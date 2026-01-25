"""
基于 Streamlit 的 SQL Agent Web 界面
提供聊天式交互和结果可视化
"""
import streamlit as st
import sys
import os
import json
from io import BytesIO
from datetime import datetime
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import BaseCallbackHandler
import time
import pandas as pd
from sqlalchemy import create_engine
import hashlib
import re
import asyncio

# Add missing import for HumanMessage
from langchain_core.messages import HumanMessage

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlagent import SQLAgent, Config
from sqlagent.agent import LIMIT_ALL, extract_limit_with_llm
from sqlagent.security import sanitize_sql_query, MAX_HARD_LIMIT
from sqlagent.code_sandbox import execute_analysis_code

# ECharts（可选依赖，未安装时自动降级不显示图表）
try:
    from streamlit_echarts import st_echarts  # type: ignore
    HAS_ECHARTS = True
except Exception:
    HAS_ECHARTS = False


class StreamlitStatusTraceHandler(BaseCallbackHandler):
    """把工具调用过程实时写到 Streamlit 的 st.status，展示完整执行流程。"""

    def __init__(self, status_box):
        self.status_box = status_box
        self.step_no = 0
        self._current_tool: Optional[str] = None

    @staticmethod
    def _norm(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, str):
            return v.strip()
        if isinstance(v, dict):
            # 常见：{"query": "..."} / {"tool_input": "..."}
            if "query" in v and isinstance(v["query"], str):
                return v["query"].strip()
            if "tool_input" in v and isinstance(v["tool_input"], str):
                return v["tool_input"].strip()
            return str(v)
        return str(v).strip()

    def on_tool_start(self, serialized: Dict[str, Any], input_str: Any = None, **kwargs) -> None:
        tool_name = (serialized or {}).get("name") or kwargs.get("name") or ""
        normalized_input = self._norm(input_str if input_str is not None else kwargs.get("input"))

        self.step_no += 1
        self._current_tool = tool_name
        
        # 显示工具调用信息
        self.status_box.write(f"**{self.step_no}. 调用工具：`{tool_name}`**")
        if normalized_input:
            if tool_name == "sql_db_query":
                self.status_box.write("输入（SQL）：")
                self.status_box.code(normalized_input, language="sql")
            else:
                self.status_box.write("输入：")
                self.status_box.code(normalized_input)

    def on_tool_end(self, output: Any, **kwargs) -> None:
        normalized_output = self._norm(output)
        if normalized_output:
            self.status_box.write("输出：")
            self.status_box.code(normalized_output)

    def on_tool_error(self, error: Exception, **kwargs) -> None:
        self.status_box.write(f"❌ 工具执行出错：{error}")


class StreamlitAnswerStreamHandler(BaseCallbackHandler):
    """只用于“最终回答”的 token 流式展示。"""

    def __init__(self, placeholder: "st.delta_generator.DeltaGenerator"):
        self.placeholder = placeholder
        self._buf: List[str] = []
        self._last_flush = 0.0

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self._buf.append(token)
        now = time.monotonic()
        # 轻微节流，避免每个 token 都触发 UI 重绘
        if now - self._last_flush >= 0.05:
            self.placeholder.markdown("".join(self._buf))
            self._last_flush = now

    def flush(self) -> str:
        text = "".join(self._buf)
        self.placeholder.markdown(text)
        return text


@st.cache_resource
def get_sqlalchemy_engine(db_name: str):
    """复用 SQLAlchemy Engine（避免每次都新建连接池）。"""
    uri = Config.get_db_uri(db_name)
    engine_args = {
        "pool_pre_ping": True,
        "pool_size": 5,
        "max_overflow": 10,
        "pool_recycle": 3600,
        "connect_args": {"connect_timeout": 10},
    }
    return create_engine(uri, **engine_args)




def build_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="result")
    return buf.getvalue()

def df_preview_text(df: pd.DataFrame, n: int = 20) -> str:
    """给模型用的样例文本：最多 n 行，避免上下文爆炸。"""
    if df is None or df.empty:
        return "(空结果)"
    d = df.head(n).copy()
    # 全部转成字符串，避免 datetime/decimal 等在 markdown 里过长
    for c in d.columns:
        d[c] = d[c].astype(str)
    try:
        return d.to_markdown(index=False)
    except Exception:
        # 兜底：CSV
        return d.to_csv(index=False)

def _chinese_num_to_int(s: str) -> Optional[int]:
    """支持 1-99 的中文数字（含“十/二十/二十一”）。"""
    s = s.strip()
    if not s:
        return None
    mapping = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if s == "十":
        return 10
    if "十" in s:
        left, _, right = s.partition("十")
        tens = 1 if left == "" else mapping.get(left)
        if tens is None:
            return None
        ones = 0 if right == "" else mapping.get(right)
        if ones is None:
            return None
        return tens * 10 + ones
    if len(s) == 1 and s in mapping:
        return mapping[s]
    return None


def _parse_user_limit(question: str) -> Optional[int]:
    """规则优先：返回用户明确要求的行数；无法识别则返回 None。"""
    q = (question or "").strip()
    if not q:
        return None

    # 阿拉伯数字
    m = re.search(r"(?i)(?:top|前|最多|只要|限制|返回|显示|取)\s*(\d{1,4})\s*(?:条|行)?", q)
    if m:
        return max(1, int(m.group(1)))

    # 中文数字
    m2 = re.search(r"(?:top|前|最多|只要|限制|返回|显示|取)\s*([一二两三四五六七八九十]{1,3})\s*(?:条|行)?", q)
    if m2:
        n2 = _chinese_num_to_int(m2.group(1))
        if n2 is not None:
            return max(1, n2)

    # 全量关键词（放在数字之后，避免“所有…前5”被误判为全量）
    if re.search(r"(全部|所有|全量|列出所有|查询所有|全部数据|所有数据)", q):
        return LIMIT_ALL

    return None


def resolve_download_limit(question: str, llm) -> Optional[int]:
    """
    规则优先 + LLM 兜底：
    - 返回 LIMIT_ALL 表示用户要全量
    - 返回 N 表示用户要 N 行
    - 返回 None 表示无法判断（按原 SQL 执行）
    """
    rule_limit = _parse_user_limit(question)
    if rule_limit is not None:
        return rule_limit

    try:
        llm_limit = extract_limit_with_llm(question, llm)
        if llm_limit:
            return llm_limit
    except Exception:
        pass

    return None

@st.cache_data(ttl=300, show_spinner=False)
def get_df_for_sql(db_name: str, sql: str) -> pd.DataFrame:
    """为历史消息重绘/下载重跑提供 DataFrame（带缓存，避免频繁打库）。"""
    engine = get_sqlalchemy_engine(db_name)
    return execute_sql_to_df(sql, engine)

def stable_key_for_sql(db_name: str, sql: str) -> str:
    raw = (db_name + "\n" + (sql or "")).encode("utf-8", errors="ignore")
    return hashlib.md5(raw).hexdigest()


# 缓存图表配置，避免历史消息重复调用 LLM
@st.cache_data(ttl=600, show_spinner=False)
def get_cached_echarts_option(cache_key: str, question: str, sql: str, df_info_json: str, _agent) -> Dict[str, Any]:
    """缓存 ECharts 配置生成结果，避免历史消息每次 rerun 都重新调用 LLM。"""
    import json
    df_info = json.loads(df_info_json) if df_info_json else {}
    return _agent.generate_echarts_option(question=question, sql=sql, df_info=df_info)

def build_df_info_for_viz(df: pd.DataFrame, max_rows: int = 20) -> Dict[str, Any]:
    """给 LLM 用的可视化上下文：避免塞全量，提供列类型/基数/样例/简单统计。"""
    if df is None or df.empty:
        return {"row_count": 0, "columns": [], "sample_rows": []}

    d = df.copy().head(max_rows)

    cols_info: List[Dict[str, Any]] = []
    for c in d.columns:
        s = d[c]
        # 基本类型
        if pd.api.types.is_numeric_dtype(s):
            col_type = "number"
        elif pd.api.types.is_datetime64_any_dtype(s):
            col_type = "datetime"
        else:
            col_type = "string"

        nunique = int(s.astype(str).nunique(dropna=True))
        cols_info.append(
            {
                "name": str(c),
                "type": col_type,
                "nunique": nunique,
            }
        )

    # 样例行：转成纯 Python 类型，避免 datetime/decimal 序列化问题
    sample_rows = d.astype(str).to_dict(orient="records")

    # 数值列简单统计
    num_cols = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
    numeric_summary: Dict[str, Any] = {}
    for c in num_cols[:5]:
        s = pd.to_numeric(d[c], errors="coerce")
        numeric_summary[str(c)] = {
            "min": float(s.min()) if s.notna().any() else None,
            "max": float(s.max()) if s.notna().any() else None,
            "mean": float(s.mean()) if s.notna().any() else None,
        }

    return {
        "row_count": int(len(df)),
        "columns": cols_info,
        "sample_rows": sample_rows,
        "numeric_summary": numeric_summary,
        "note": f"sample_rows 仅为前 {max_rows} 行，用于选图；真实结果行数见 row_count。",
    }

def execute_sql_to_df(sql: str, engine) -> pd.DataFrame:
    """
    使用 SQLAlchemy 的 raw_connection 获取 DBAPI 连接直接执行 SQL，
    规避某些驱动在包含 LIKE '%xx%' 时把 % 误当作占位符导致的格式化错误。
    """
    q = (sql or "").strip().rstrip(";")
    if not q:
        return pd.DataFrame()

    conn = engine.raw_connection()
    try:
        cur = conn.cursor()
        cur.execute(q)  # 不传参数，确保 % 作为 SQL 字面量生效
        rows = cur.fetchall()
        cols = [d[0] for d in (cur.description or [])]
        return pd.DataFrame(list(rows), columns=cols)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def execute_scalar(sql: str, engine) -> Any:
    """执行返回单个值的 SQL（例如 COUNT(*)）。"""
    q = (sql or "").strip().rstrip(";")
    if not q:
        return None
    conn = engine.raw_connection()
    try:
        cur = conn.cursor()
        cur.execute(q)
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def strip_trailing_limit(sql: str) -> str:
    """
    去除末尾 LIMIT 子句（仅处理末尾的 LIMIT n / LIMIT offset,n / LIMIT n OFFSET offset）。
    用于计算“全量行数”COUNT(*)。
    """
    s = (sql or "").strip().rstrip(";")
    # 移除末尾 LIMIT ...（尽量不影响子查询内 LIMIT）
    s = re.sub(r"(?is)\s+limit\s+\d+\s*,\s*\d+\s*$", "", s)
    s = re.sub(r"(?is)\s+limit\s+\d+\s+offset\s+\d+\s*$", "", s)
    s = re.sub(r"(?is)\s+limit\s+\d+\s*$", "", s)
    return s.strip()


def auto_echarts_option(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """根据 df 的列类型自动选择一个合适的 ECharts option（简单规则版）。"""
    if df is None or df.empty:
        return None

    # 绘图最多使用 20 行（与结果限制一致）
    d = df.copy().head(20)

    # 尝试识别时间列
    datetime_cols: List[str] = []
    for c in d.columns:
        if pd.api.types.is_datetime64_any_dtype(d[c]):
            datetime_cols.append(c)
            continue
        if pd.api.types.is_object_dtype(d[c]):
            # 尝试 parse
            parsed = pd.to_datetime(d[c], errors="coerce", utc=False)
            if parsed.notna().mean() > 0.8:
                d[c] = parsed
                datetime_cols.append(c)

    numeric_cols = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
    cat_cols = [c for c in d.columns if c not in numeric_cols and c not in datetime_cols]

    # 时间 + 数值 => 折线
    if datetime_cols and numeric_cols:
        x = datetime_cols[0]
        y = numeric_cols[0]
        dd = d[[x, y]].dropna().sort_values(x)
        return {
            "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "category", "data": dd[x].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()},
            "yAxis": {"type": "value"},
            "series": [{"type": "line", "data": dd[y].tolist(), "smooth": True}],
        }

    # 分类 + 数值 => 条形
    if cat_cols and numeric_cols:
        x = cat_cols[0]
        y = numeric_cols[0]
        dd = d[[x, y]].dropna()
        # 取前 20 类
        dd = dd.head(20)
        return {
            "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "category", "data": dd[x].astype(str).tolist(), "axisLabel": {"rotate": 30}},
            "yAxis": {"type": "value"},
            "series": [{"type": "bar", "data": dd[y].tolist()}],
        }

    # 两个数值 => 散点
    if len(numeric_cols) >= 2:
        x, y = numeric_cols[0], numeric_cols[1]
        dd = d[[x, y]].dropna().head(500)
        return {
            "tooltip": {"trigger": "item"},
            "xAxis": {"type": "value", "name": x},
            "yAxis": {"type": "value", "name": y},
            "series": [{"type": "scatter", "data": dd.values.tolist()}],
        }

    return None

# 页面配置
st.set_page_config(
    page_title="SQL Agent - 智能查询助手",
    page_icon="🤖",
    layout="wide"
)

# 自定义样式：减少闪烁，增强视觉反馈
st.markdown("""
<style>
    /* 进度提示动画 */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    .stStatus { transition: all 0.3s ease; }
    
    /* 数据表格过渡 */
    .stDataFrame { 
        animation: fadeIn 0.3s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* 图表容器 */
    iframe[title*="streamlit_echarts"] {
        animation: slideUp 0.4s ease-out;
    }
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* 聊天消息优化 */
    .stChatMessage {
        transition: all 0.2s ease;
    }
    
    /* 下载按钮悬停效果 */
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .stDownloadButton > button {
        transition: all 0.2s ease;
    }
    
    /* 侧边栏历史对话按钮样式 */
    
    /* 关键技巧：利用 :has() 选择器，当水平块内包含 primary 按钮（选中状态）时，
       给整个水平块（HorizontalBlock）添加背景色。
    */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(button[kind="primary"]) {
        background-color: rgba(26, 115, 232, 0.15);
        border-radius: 6px;
        padding-left: 4px;
        transition: all 0.2s ease-in-out; /* 丝滑过渡 */
    }

    /* 悬停未选中行时：整体背景变灰 */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(button[kind="secondary"]):hover {
        background-color: #f0f0f0;
        border-radius: 6px;
        padding-left: 4px;
        transition: all 0.2s ease-in-out; /* 丝滑过渡 */
    }
    
    /* 选中状态（primary 按钮）：背景透明，文字蓝色 */
    [data-testid="stSidebar"] .stButton button[kind="primary"] {
        background-color: transparent !important;
        color: #1a73e8 !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    [data-testid="stSidebar"] .stButton button[kind="primary"]:hover {
        background-color: transparent !important;
        color: #1a73e8 !important;
    }
    
    /* 未选中状态（secondary 按钮） */
    [data-testid="stSidebar"] .stButton button[kind="secondary"] {
        background-color: transparent !important;
        color: inherit !important;
        border: none !important; /* 去掉边框 */
        transition: color 0.15s ease;
    }
    
    /* 未选中状态悬停按钮：不单独变色，由父容器变色 */
    [data-testid="stSidebar"] .stButton button[kind="secondary"]:hover {
        background-color: transparent !important;
        color: inherit !important;
        border-color: transparent !important;
    }
    
    /* 三点菜单按钮 */
    [data-testid="stSidebar"] .stPopover button {
        padding: 4px 8px !important;
        min-height: auto !important;
        background: transparent !important;
        border: none !important;
        opacity: 0; /* 默认隐藏 */
        transition: opacity 0.2s ease-in-out; /* 丝滑显隐 */
    }
    
    /* 选中行时：显示三点菜单 */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(button[kind="primary"]) .stPopover button {
        opacity: 1;
    }
    
    /* 悬停行时（无论选中与否）：显示三点菜单 */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:hover .stPopover button {
        opacity: 1;
    }
    
    [data-testid="stSidebar"] .stPopover button:hover {
        background: rgba(0,0,0,0.05) !important;
    }
</style>
""", unsafe_allow_html=True)

# 使用 @st.cache_resource 缓存 Agent 对象（跨会话共享，刷新页面不重新初始化）
@st.cache_resource
def get_sql_agent(db_name: str = None):
    """
    获取缓存的 SQL Agent 对象
    使用 @st.cache_resource 使连接在所有用户会话间共享
    刷新页面或新标签页都不会重新初始化
    """
    return SQLAgent(db_name=db_name)

# 初始化 session state
if "conversations" not in st.session_state:
    st.session_state.conversations = []
if "current_conv_id" not in st.session_state:
    st.session_state.current_conv_id = None
if "db_name" not in st.session_state:
    st.session_state.db_name = Config.DB_NAME

# 确保至少有一个对话
if not st.session_state.conversations:
    conv_id = f"conv-{int(time.time() * 1000)}"
    now_ts = time.time()
    st.session_state.conversations.append(
        {"id": conv_id, "messages": [], "created_at": now_ts, "updated_at": now_ts}
    )
    st.session_state.current_conv_id = conv_id

# 绑定当前对话的消息列表
current_conv = next(
    (c for c in st.session_state.conversations if c["id"] == st.session_state.current_conv_id),
    None
)
if current_conv is None:
    conv_id = f"conv-{int(time.time() * 1000)}"
    now_ts = time.time()
    current_conv = {"id": conv_id, "messages": [], "created_at": now_ts, "updated_at": now_ts}
    st.session_state.conversations.append(current_conv)
    st.session_state.current_conv_id = conv_id

st.session_state.messages = current_conv["messages"]

# 用于更顺滑的“运行中禁用输入框”体验（两段式 rerun）
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "is_running" not in st.session_state:
    st.session_state.is_running = False

# 添加数据分析报告相关的session state
if "analysis_reports" not in st.session_state:
    st.session_state.analysis_reports = {}

if "current_analysis_df" not in st.session_state:
    st.session_state.current_analysis_df = None

if "current_analysis_question" not in st.session_state:
    st.session_state.current_analysis_question = ""

# 获取缓存的 Agent（首次加载会显示加载提示）
try:
    with st.spinner("正在连接数据库..."):
        agent = get_sql_agent(st.session_state.db_name)
except Exception as e:
    st.error(f"初始化 Agent 失败: {e}")
    st.stop()

# 侧边栏配置
with st.sidebar:
    # 新建对话（放在侧边栏顶部）
    if st.button("🆕 新建对话", use_container_width=True, disabled=not st.session_state.messages):
        conv_id = f"conv-{int(time.time() * 1000)}"
        now_ts = time.time()
        st.session_state.conversations.append(
            {"id": conv_id, "messages": [], "created_at": now_ts, "updated_at": now_ts}
        )
        st.session_state.current_conv_id = conv_id
        st.rerun()

    # 历史对话（GPT 风格列表）
    st.subheader("🕘 历史对话")

    def _conv_label(conv: Dict[str, Any]) -> str:
        for msg in conv.get("messages", []):
            if msg.get("role") == "user" and msg.get("content"):
                text = msg.get("content", "").strip()
                if text:
                    return text[:20] + ("..." if len(text) > 20 else "")
        return "新对话"

    sorted_convs = sorted(
        st.session_state.conversations,
        key=lambda c: c.get("updated_at", c.get("created_at", 0)),
        reverse=True,
    )
    
    # 重命名状态
    if "renaming_conv_id" not in st.session_state:
        st.session_state.renaming_conv_id = None
    
    if sorted_convs:
        for conv in sorted_convs:
            conv_id = conv.get("id")
            label = conv.get("title") or _conv_label(conv)
            is_selected = conv_id == st.session_state.current_conv_id
            
            # 如果正在重命名这个对话
            if st.session_state.renaming_conv_id == conv_id:
                col_input, col_ok = st.columns([0.8, 0.2])
                with col_input:
                    new_title = st.text_input(
                        "重命名",
                        value=label,
                        key=f"rename-input-{conv_id}",
                        label_visibility="collapsed",
                    )
                with col_ok:
                    if st.button("✓", key=f"rename-ok-{conv_id}"):
                        conv["title"] = new_title
                        st.session_state.renaming_conv_id = None
                        st.rerun()
            else:
                # 正常显示：对话按钮 + 三点菜单
                col_btn, col_menu = st.columns([0.85, 0.15])
                with col_btn:
                    if st.button(
                        label,
                        key=f"conv-{conv_id}",
                        use_container_width=True,
                        type="primary" if is_selected else "secondary",
                    ):
                        st.session_state.current_conv_id = conv_id
                        st.rerun()
                
                # 悬停或选中时显示三点菜单（目前 Streamlit 只能实现选中时一直显示，或者一直显示）
                # 为了实现"悬停显示"效果，我们需要接受三点菜单一直存在，但未选中时颜色淡化
                with col_menu:
                    # 使用 popover 并自定义 CSS 让其默认透明，悬停显色（较难完美实现）
                    # 现在的妥协方案：一直显示三点菜单，但未选中时淡化
                    with st.popover("⋮", use_container_width=True):
                        if st.button("✏️ 重命名", key=f"rename-{conv_id}", use_container_width=True):
                            st.session_state.renaming_conv_id = conv_id
                            st.rerun()
                        if st.button("🗑️ 删除", key=f"delete-{conv_id}", use_container_width=True):
                            st.session_state.conversations = [
                                c for c in st.session_state.conversations if c["id"] != conv_id
                            ]
                            # 如果删除的是当前对话，切换到最新的
                            if st.session_state.current_conv_id == conv_id:
                                remaining = st.session_state.conversations
                                if remaining:
                                    st.session_state.current_conv_id = remaining[-1]["id"]
                                else:
                                    # 没有对话了，创建一个新的
                                    new_id = f"conv-{int(time.time() * 1000)}"
                                    now_ts = time.time()
                                    st.session_state.conversations.append(
                                        {"id": new_id, "messages": [], "created_at": now_ts, "updated_at": now_ts}
                                    )
                                    st.session_state.current_conv_id = new_id
                            st.rerun()
    else:
        st.caption("暂无历史对话")

    st.divider()

    # 数据库选择（固定列表，映射到实际数据库）
    st.title("⚙️ 配置")
    
    # 显示名称 -> 实际数据库名 的映射
    DB_DISPLAY_MAP = {
        "Financial Asset Management": "wutongbei",
        "Healthcare Analytics": "wutongbei",  # 假选项，实际也连到 wutongbei
    }
    display_names = list(DB_DISPLAY_MAP.keys())
    
    # 反向映射：实际数据库名 -> 显示名称（用于显示当前选中项）
    def get_display_name(actual_db: str) -> str:
        for name, db in DB_DISPLAY_MAP.items():
            if db == actual_db:
                return name
        return actual_db
    
    try:
        current_display = get_display_name(st.session_state.db_name)
        current_index = display_names.index(current_display) if current_display in display_names else 0
        
        selected_display = st.selectbox(
            "选择数据库",
            display_names,
            index=current_index
        )
        
        # 将显示名称映射为实际数据库名
        actual_db = DB_DISPLAY_MAP.get(selected_display, "wutongbei")

        if actual_db != st.session_state.db_name:
            with st.spinner(f"切换到数据库 {selected_display}..."):
                # 清除缓存，重新获取新数据库的 Agent
                get_sql_agent.clear()
                st.session_state.db_name = actual_db
                st.rerun()  # 重新运行以使用新的数据库
    except Exception as e:
        st.error(f"数据库选择失败: {e}")

    st.divider()

    # 已移除清空对话按钮，改为“新建对话”

# 主界面
st.title("🤖 SQL Agent - 智能 MySQL 查询助手")
# 显示友好的数据库名称
_db_friendly_names = {"wutongbei": "Financial Asset Management"}
_current_db_display = _db_friendly_names.get(st.session_state.db_name, st.session_state.db_name)
st.caption(f"当前数据库: **{_current_db_display}**")

# 显示对话历史（使用缓存的数据，避免重复渲染）
for msg_idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        # 如果是助手消息且有工具调用轨迹，先显示轨迹
        if message["role"] == "assistant" and message.get("trace"):
            with st.expander("🔧 查看工具调用过程", expanded=False):
                for step_idx, step in enumerate(message["trace"], 1):
                    tool_name = step.get("tool", "")
                    tool_input = step.get("input", "")
                    tool_output = step.get("output", "")
                    st.write(f"**{step_idx}. 调用工具：`{tool_name}`**")
                    if tool_input:
                        if tool_name == "sql_db_query":
                            st.write("输入（SQL）：")
                            st.code(tool_input, language="sql")
                        else:
                            st.write("输入：")
                            st.code(tool_input)
                    if tool_output:
                        st.write("输出：")
                        st.code(tool_output[:2000] + ("..." if len(tool_output) > 2000 else ""))
        
        # 如果是助手消息且有SQL，显示SQL（优先展示下载SQL）
        if message["role"] == "assistant" and ("download_sql" in message or "sql" in message):
            with st.expander("📝 查看生成的 SQL 语句", expanded=False):
                st.code(message.get("download_sql") or message.get("sql", ""), language="sql")

        # 如果历史消息里有 last_sql，则重绘"表格 + 下载 + 图表"
        if message["role"] == "assistant" and message.get("last_sql"):
            try:
                # 获取缓存的 full_count
                full_count = message.get("full_count")

                # 预览用 SQL（固定最多50行）：先去掉原 LIMIT，再强制加 LIMIT 50
                preview_limit = 50
                sql_no_limit = strip_trailing_limit(message["last_sql"])
                preview_sql = sanitize_sql_query(sql_no_limit, default_limit=preview_limit, hard_limit=preview_limit)
                df_preview = get_df_for_sql(st.session_state.db_name, preview_sql)
                
                # 显示全量行数
                if full_count is not None and full_count <= preview_limit:
                    st.markdown(f"**📄 查询结果（共 {full_count} 行，已全部展示）**")
                elif full_count is not None:
                    st.markdown(f"**📄 查询结果（前 {len(df_preview)} 行 / 共 {full_count} 行）**")
                elif len(df_preview) <= preview_limit:
                    st.markdown(f"**📄 查询结果（共 {len(df_preview)} 行，已全部展示）**")
                else:
                    st.markdown(f"**📄 查询结果（前 {len(df_preview)} 行）**")
                st.dataframe(df_preview, use_container_width=True)

                # 下载：按用户意图决定行数（规则优先 + LLM 兜底）
                question_text = message.get("question", "") or ""
                intent_limit = resolve_download_limit(question_text, agent.llm)
                if intent_limit is None:
                    download_sql = message["last_sql"]
                elif intent_limit >= LIMIT_ALL:
                    download_sql = strip_trailing_limit(message["last_sql"])
                else:
                    download_sql = sanitize_sql_query(
                        message["last_sql"], default_limit=intent_limit, hard_limit=intent_limit
                    )
                df_download = get_df_for_sql(st.session_state.db_name, download_sql)
                excel_bytes = build_excel_bytes(df_download)
                filename = f"query_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                dl_key = (
                    f"download-{stable_key_for_sql(st.session_state.db_name, message['last_sql'])}"
                    f"-msg-{msg_idx}"
                )
                download_rows = len(df_download)
                st.download_button(
                    label=f"⬇️ 下载结果（Excel，共 {download_rows} 行）",
                    data=excel_bytes,
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=dl_key,
                )

                # 新增：自定义分析要求输入框
                custom_analysis_req = st.text_input(
                    "📊 自定义分析要求（可选）:",
                    key=f"custom-analysis-req-{dl_key}",
                    placeholder="例如：重点关注销售额趋势、按地区分组统计、生成相关性热力图等"
                )

                # 新增：生成数据分析报告按钮
                report_key = f"report-{dl_key}"
                if st.button("📊 生成数据分析报告", key=f"generate-report-{dl_key}"):
                    try:
                        with st.spinner("正在生成数据分析报告..."):
                            # 保存当前分析所需的数据
                            st.session_state.current_analysis_df = df_download.copy()
                            st.session_state.current_analysis_question = message.get("question", "")
                            
                            # 生成Excel文件用于沙箱执行
                            excel_filename = f"temp_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                            excel_path = f"/tmp/{excel_filename}"
                            df_download.to_excel(excel_path, index=False)
                            
                            # 生成PDF报告文件路径
                            report_filename = f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                            report_path = f"/tmp/{report_filename}"
                            
                            # 构建分析提示词
                            base_prompt = f"""
你是一名专业的数据科学家。请基于用户的问题和提供的DataFrame（变量名: df）生成完整的Python数据分析代码。

用户问题: {st.session_state.current_analysis_question}
"""
                            
                            if custom_analysis_req.strip():
                                base_prompt += f"\n用户自定义分析要求: {custom_analysis_req.strip()}\n"
                            
                            analysis_prompt = base_prompt + f"""
要求：
1. 进行全面的数据分析，包括基本统计、相关性分析、分布分析等
2. 生成高质量的可视化图表（使用matplotlib/seaborn），**所有图表标题、标签、图例必须使用英文**
3. 将所有分析结果整合到一个PDF报告中，**包含详细的中文文字描述和解释**
4. PDF报告必须保存到路径: {report_path}
5. **中文文字描述内容应包括**：
   - 数据概览（形状、列信息、数据类型）
   - 缺失值分析
   - 基本统计摘要
   - 数值列的分布特征分析
   - 分类列的频次分析  
   - 相关性分析（如果适用）
   - 关键发现和洞察总结
6. 不要使用plt.show()，直接保存图表到PDF
7. 确保代码完整可运行，包含所有必要的import语句
8. **重要：处理中文标签时使用提供的转换函数**：
   ```python
   # 如果您的数据包含中文分类标签，使用以下函数转换为英文
   # x_labels = safe_translate_labels(original_labels)
   # ax.set_xticklabels(x_labels)
   
   # 或者在创建图表时直接转换
   # categories = [translate_chinese_to_english(cat) for cat in original_categories]
   ```
9. **必须包含matplotlib英文配置**：
   ```python
   import matplotlib.pyplot as plt
   import matplotlib
   matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
   matplotlib.rcParams['axes.unicode_minus'] = False
   ```
10. **必须包含reportlab中文字体配置**：
    ```python
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import os
    
    # 注册中文字体用于PDF文字描述
    font_name = 'Helvetica'
    try:
        font_paths_to_try = [
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        ]
        
        for font_path in font_paths_to_try:
            try:
                if os.path.exists(font_path):
                    if font_path.endswith('.ttc'):
                        pdfmetrics.registerFont(TTFont('ChineseFont', font_path))
                    else:
                        pdfmetrics.registerFont(TTFont('ChineseFont', font_path))
                    font_name = 'ChineseFont'
                    break
            except Exception:
                continue
    except Exception:
        pass  # 使用默认字体
    ```
11. 在创建PDF样式时，使用 fontName=font_name 参数
12. 确保所有图表在保存前调用 plt.savefig()，保存后调用 plt.close()

**数据特点说明：**
- 数据可能包含中文列名和文本数据
- 包含日期时间列（register_date, create_time, update_time）
- 数值列可能包括total_assets
- 需要适当处理非数值列，避免在数值分析中出错

**输出格式要求：**
- PDF报告应该包含中文文字描述段落和对应的英文图表
- 每个分析部分都应该有清晰的中文标题和详细的文字解释
- 文字描述应该专业、准确、易于理解
- **图表中的所有文字（包括坐标轴标签、图例、标题）必须是英文**

请只返回Python代码，不要包含任何解释性文字。
"""
                            
                            # 调用大模型生成分析代码
                            analysis_code_response = agent.llm.invoke([HumanMessage(content=analysis_prompt)])
                            analysis_code = analysis_code_response.content.strip()
                            
                            # 如果代码被包裹在代码块中，提取纯代码
                            if analysis_code.startswith("```python"):
                                analysis_code = analysis_code[9:-3] if analysis_code.endswith("```") else analysis_code[9:]
                            elif analysis_code.startswith("```"):
                                analysis_code = analysis_code[3:-3] if analysis_code.endswith("```") else analysis_code[3:]
                            
                            # 在沙箱中执行代码
                            result = asyncio.run(
                                execute_analysis_code(analysis_code, excel_path, report_path, timeout=120)
                            )
                            
                            if result["success"] and os.path.exists(report_path):
                                # 读取PDF文件
                                with open(report_path, "rb") as f:
                                    pdf_bytes = f.read()
                                    
                                # 保存到session state以便下载
                                report_cache_key = stable_key_for_sql(st.session_state.db_name, message['last_sql'])
                                st.session_state.analysis_reports[report_cache_key] = {
                                    'pdf_bytes': pdf_bytes,
                                    'filename': report_filename
                                }
                                
                                st.success("✅ 数据分析报告生成成功！")
                                st.download_button(
                                    label="📥 下载数据分析报告 (PDF)",
                                    data=pdf_bytes,
                                    file_name=report_filename,
                                    mime="application/pdf",
                                    key=f"download-report-{dl_key}"
                                )
                            else:
                                error_msg = result.get("error", "未知错误")
                                st.error(f"❌ 报告生成失败: {error_msg}")
                                # 显示生成的代码以便调试
                                with st.expander("查看生成的代码（用于调试）"):
                                    st.code(analysis_code, language="python")
                    
                    except Exception as e:
                        st.error(f"❌ 报告生成出错: {str(e)}")
                        import traceback
                        st.text(traceback.format_exc())
                    finally:
                        # 清理临时文件
                        try:
                            if 'excel_path' in locals() and os.path.exists(excel_path):
                                os.remove(excel_path)
                            if 'report_path' in locals() and os.path.exists(report_path):
                                os.remove(report_path)
                        except Exception:
                            pass

                # 图表渲染：优先使用已缓存的 echarts_option，避免重复调用 LLM
                if HAS_ECHARTS:
                    cached_viz = message.get("echarts_viz")
                    if cached_viz is not None:
                        # 直接使用已存储的配置
                        if cached_viz.get("show") and isinstance(cached_viz.get("option"), dict):
                            st.markdown("**📊 可视化（ECharts）**")
                            st_echarts(cached_viz["option"], height="420px", key=f"chart-{dl_key}")
                        elif cached_viz.get("reason"):
                            st.caption(f"📊 不展示图表：{cached_viz.get('reason', '数据不适合')}")
                    else:
                        # 兜底：使用缓存函数（仍然可能调用 LLM，但有 TTL 缓存）
                        try:
                            df_info = build_df_info_for_viz(df_preview, max_rows=20)
                            cache_key = f"hist-{dl_key}"
                            viz = get_cached_echarts_option(
                                cache_key=cache_key,
                                question="(历史消息重绘)",
                                sql=message.get("last_sql", ""),
                                df_info_json=json.dumps(df_info, ensure_ascii=False, default=str),
                                _agent=agent,
                            )
                            if viz.get("show") and isinstance(viz.get("option"), dict):
                                st.markdown("**📊 可视化（ECharts）**")
                                st_echarts(viz["option"], height="420px", key=f"chart-{dl_key}")
                            else:
                                st.caption(f"📊 不展示图表：{viz.get('reason', '数据不适合')}")
                        except Exception as e:
                            st.caption(f"📊 图表生成失败：{e}")
            except Exception as e:
                st.caption(f"⚠️ 查询结果展示失败：{e}")
        
        # 显示 Token 使用统计（历史消息，美化版）
        if message["role"] == "assistant" and message.get("token_usage"):
            token_usage = message["token_usage"]
            input_tokens = token_usage.get("input_tokens", 0)
            output_tokens = token_usage.get("output_tokens", 0)
            total_tokens = token_usage.get("total_tokens", 0) or (input_tokens + output_tokens)
            llm_calls = token_usage.get("llm_calls", 0)
            if total_tokens > 0:
                st.markdown("---")
                st.markdown("##### 📊 Token 使用统计")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📥 输入", f"{input_tokens:,}")
                with col2:
                    st.metric("📤 输出", f"{output_tokens:,}")
                with col3:
                    st.metric("📊 总计", f"{total_tokens:,}")
                with col4:
                    st.metric("🔄 调用次数", f"{llm_calls}")
        
        # 显示消息内容（允许助手消息 content 为空：只展示表格/下载等组件，不额外输出"查询完成…"文案）
        content = message.get("content", "")
        if isinstance(content, str) and content.strip():
            st.markdown(content)

# 输入框
# 输入框（运行中禁用，并提示“正在运行”）
prompt_input = st.chat_input(
    "请输入您的问题..." if not st.session_state.is_running else "正在查询中，请稍候…",
    disabled=bool(st.session_state.is_running),
)

# 第一步：用户提交后先缓存 prompt 并 rerun，使输入框立即进入“运行中”状态（更顺滑）
if prompt_input:
    st.session_state.pending_prompt = prompt_input
    st.session_state.is_running = True
    st.rerun()

# 第二步：如果有待处理 prompt 且 is_running=True，就执行查询
if st.session_state.pending_prompt and st.session_state.is_running:
    prompt = st.session_state.pending_prompt

    # 添加用户消息（只添加一次）
    st.session_state.messages.append({"role": "user", "content": prompt})
    current_conv["updated_at"] = time.time()
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 工具调用流程展示（默认折叠，用户点击可展开查看详情）
        status_box = st.status("正在执行查询...", expanded=False, state="running")
        live_handler = StreamlitStatusTraceHandler(status_box)

        tool_result = agent.run_tools(prompt, callbacks=[live_handler])

        if tool_result.get("success"):
            status_box.update(label="查询执行成功（点击展开详情）", state="complete", expanded=False)
        else:
            status_box.update(label="查询执行失败（点击查看详情）", state="error", expanded=False)

        df = None
        full_count = None
        echarts_viz = None  # 用于缓存到消息

        if tool_result.get("success"):
            if tool_result.get("sql"):
                # 优先展示按用户意图生成的“下载 SQL”
                display_sql = None
                if "download_sql" in locals():
                    display_sql = download_sql
                with st.expander("📝 查看生成的 SQL 语句", expanded=False):
                    st.code(display_sql or tool_result["sql"], language="sql")

            last_sql = tool_result.get("last_sql", "") or ""
            if last_sql:
                try:
                    engine = get_sqlalchemy_engine(st.session_state.db_name)
                    
                    # 计算"全量行数"：对去掉 LIMIT 的 SQL 做 COUNT(*)
                    try:
                        sql_no_limit = strip_trailing_limit(last_sql)
                        count_sql = f"SELECT COUNT(*) FROM ({sql_no_limit}) AS t"
                        full_count = int(execute_scalar(count_sql, engine))
                    except Exception:
                        full_count = None
                    
                    # 预览用 SQL（固定最多50行）：先去掉原 LIMIT，再强制加 LIMIT 50
                    preview_limit = 50
                    sql_no_limit_preview = strip_trailing_limit(last_sql)
                    preview_sql = sanitize_sql_query(sql_no_limit_preview, default_limit=preview_limit, hard_limit=preview_limit)
                    df_preview = execute_sql_to_df(preview_sql, engine)

                    # 显示预览表格
                    if full_count is not None and full_count <= preview_limit:
                        st.markdown(f"**📄 查询结果（共 {full_count} 行，已全部展示）**")
                    elif full_count is not None:
                        st.markdown(f"**📄 查询结果（前 {len(df_preview)} 行 / 共 {full_count} 行）**")
                    else:
                        st.markdown(f"**📄 查询结果（前 {len(df_preview)} 行）**")
                    st.dataframe(df_preview, use_container_width=True)

                    # 下载：按用户意图决定行数（规则优先 + LLM 兜底）
                    intent_limit = resolve_download_limit(prompt, agent.llm)
                    if intent_limit is None:
                        download_sql = last_sql
                    elif intent_limit >= LIMIT_ALL:
                        download_sql = strip_trailing_limit(last_sql)
                    else:
                        download_sql = sanitize_sql_query(
                            last_sql, default_limit=intent_limit, hard_limit=intent_limit
                        )
                    df_download = execute_sql_to_df(download_sql, engine)
                    excel_bytes = build_excel_bytes(df_download)
                    filename = f"query_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                    download_rows = len(df_download)
                    st.download_button(
                        label=f"⬇️ 下载结果（Excel，共 {download_rows} 行）",
                        data=excel_bytes,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"download-live-{stable_key_for_sql(st.session_state.db_name, download_sql)}",
                    )

                    # 新增：自定义分析要求输入框（实时查询结果）
                    custom_analysis_req_live = st.text_input(
                        "📊 自定义分析要求（可选）:",
                        key=f"custom-analysis-req-live-{stable_key_for_sql(st.session_state.db_name, download_sql)}",
                        placeholder="例如：重点关注销售额趋势、按地区分组统计、生成相关性热力图等"
                    )

                    # 新增：生成数据分析报告按钮（实时查询结果）
                    live_report_key = f"live-report-{stable_key_for_sql(st.session_state.db_name, download_sql)}"
                    if st.button("📊 生成数据分析报告", key=f"generate-live-report-{stable_key_for_sql(st.session_state.db_name, download_sql)}"):
                        try:
                            with st.spinner("正在生成数据分析报告..."):
                                # 保存当前分析所需的数据
                                st.session_state.current_analysis_df = df_download.copy()
                                st.session_state.current_analysis_question = prompt
                                
                                # 生成Excel文件用于沙箱执行
                                excel_filename = f"temp_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                                excel_path = f"/tmp/{excel_filename}"
                                df_download.to_excel(excel_path, index=False)
                                
                                # 生成PDF报告文件路径
                                report_filename = f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                                report_path = f"/tmp/{report_filename}"
                                
                                # 构建分析提示词
                                base_prompt = f"""
你是一名专业的数据科学家。请基于用户的问题和提供的DataFrame（变量名: df）生成完整的Python数据分析代码。

用户问题: {prompt}
"""
                                
                                if custom_analysis_req_live.strip():
                                    base_prompt += f"\n用户自定义分析要求: {custom_analysis_req_live.strip()}\n"
                                
                                analysis_prompt = base_prompt + f"""
要求：
1. 进行全面的数据分析，包括基本统计、相关性分析、分布分析等
2. 生成高质量的可视化图表（使用matplotlib/seaborn），**所有图表标题、标签、图例必须使用英文**
3. 将所有分析结果整合到一个PDF报告中，**包含详细的中文文字描述和解释**
4. PDF报告必须保存到路径: {report_path}
5. **中文文字描述内容应包括**：
   - 数据概览（形状、列信息、数据类型）
   - 缺失值分析
   - 基本统计摘要
   - 数值列的分布特征分析
   - 分类列的频次分析  
   - 相关性分析（如果适用）
   - 关键发现和洞察总结
6. 不要使用plt.show()，直接保存图表到PDF
7. 确保代码完整可运行，包含所有必要的import语句
8. **重要：处理中文标签时使用提供的转换函数**：
   ```python
   # 如果您的数据包含中文分类标签，使用以下函数转换为英文
   # x_labels = safe_translate_labels(original_labels)
   # ax.set_xticklabels(x_labels)
   
   # 或者在创建图表时直接转换
   # categories = [translate_chinese_to_english(cat) for cat in original_categories]
   ```
9. **必须包含matplotlib英文配置**：
   ```python
   import matplotlib.pyplot as plt
   import matplotlib
   matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
   matplotlib.rcParams['axes.unicode_minus'] = False
   ```
10. **必须包含reportlab中文字体配置**：
    ```python
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import os
    
    # 注册中文字体用于PDF文字描述
    font_name = 'Helvetica'
    try:
        font_paths_to_try = [
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        ]
        
        for font_path in font_paths_to_try:
            try:
                if os.path.exists(font_path):
                    if font_path.endswith('.ttc'):
                        pdfmetrics.registerFont(TTFont('ChineseFont', font_path))
                    else:
                        pdfmetrics.registerFont(TTFont('ChineseFont', font_path))
                    font_name = 'ChineseFont'
                    break
            except Exception:
                continue
    except Exception:
        pass  # 使用默认字体
    ```
11. 在创建PDF样式时，使用 fontName=font_name 参数
12. 确保所有图表在保存前调用 plt.savefig()，保存后调用 plt.close()

**数据特点说明：**
- 数据可能包含中文列名和文本数据
- 包含日期时间列（register_date, create_time, update_time）
- 数值列可能包括total_assets
- 需要适当处理非数值列，避免在数值分析中出错

**输出格式要求：**
- PDF报告应该包含中文文字描述段落和对应的英文图表
- 每个分析部分都应该有清晰的中文标题和详细的文字解释
- 文字描述应该专业、准确、易于理解
- **图表中的所有文字（包括坐标轴标签、图例、标题）必须是英文**

请只返回Python代码，不要包含任何解释性文字。
"""
                                
                                # 调用大模型生成分析代码
                                analysis_code_response = agent.llm.invoke([HumanMessage(content=analysis_prompt)])
                                analysis_code = analysis_code_response.content.strip()
                                
                                # 如果代码被包裹在代码块中，提取纯代码
                                if analysis_code.startswith("```python"):
                                    analysis_code = analysis_code[9:-3] if analysis_code.endswith("```") else analysis_code[9:]
                                elif analysis_code.startswith("```"):
                                    analysis_code = analysis_code[3:-3] if analysis_code.endswith("```") else analysis_code[3:]
                                
                                # 在沙箱中执行代码
                                result = asyncio.run(
                                    execute_analysis_code(analysis_code, excel_path, report_path, timeout=120)
                                )
                                
                                if result["success"] and os.path.exists(report_path):
                                    # 读取PDF文件
                                    with open(report_path, "rb") as f:
                                        pdf_bytes = f.read()
                                        
                                    # 保存到session state以便下载
                                    report_cache_key = stable_key_for_sql(st.session_state.db_name, download_sql)
                                    st.session_state.analysis_reports[report_cache_key] = {
                                        'pdf_bytes': pdf_bytes,
                                        'filename': report_filename
                                    }
                                    
                                    st.success("✅ 数据分析报告生成成功！")
                                    st.download_button(
                                        label="📥 下载数据分析报告 (PDF)",
                                        data=pdf_bytes,
                                        file_name=report_filename,
                                        mime="application/pdf",
                                        key=f"download-live-report-{stable_key_for_sql(st.session_state.db_name, download_sql)}"
                                    )
                                else:
                                    error_msg = result.get("error", "未知错误")
                                    st.error(f"❌ 报告生成失败: {error_msg}")
                                    # 显示生成的代码以便调试
                                    with st.expander("查看生成的代码（用于调试）"):
                                        st.code(analysis_code, language="python")
                        
                        except Exception as e:
                            st.error(f"❌ 报告生成出错: {str(e)}")
                            import traceback
                            st.text(traceback.format_exc())
                        finally:
                            # 清理临时文件
                            try:
                                if 'excel_path' in locals() and os.path.exists(excel_path):
                                    os.remove(excel_path)
                                if 'report_path' in locals() and os.path.exists(report_path):
                                    os.remove(report_path)
                            except Exception:
                                pass

                    # 可视化生成（使用预览数据）
                    if HAS_ECHARTS:
                        chart_placeholder = st.empty()
                        chart_placeholder.info("正在生成可视化图表...")
                        
                        try:
                            df_info = build_df_info_for_viz(df_preview, max_rows=20)
                            viz = agent.generate_echarts_option(
                                question=prompt,
                                sql=tool_result.get("sql", ""),
                                df_info=df_info,
                            )
                            echarts_viz = viz  # 缓存到消息
                            
                            chart_placeholder.empty()  # 清除 loading 提示
                            if viz.get("show") and isinstance(viz.get("option"), dict):
                                st.markdown("**📊 可视化（ECharts）**")
                                st_echarts(viz["option"], height="420px", key=f"chart-live-{stable_key_for_sql(st.session_state.db_name, preview_sql)}")
                            else:
                                st.caption(f"📊 不展示图表：{viz.get('reason', '数据不适合')}")
                        except Exception as e:
                            chart_placeholder.empty()
                            st.caption(f"📊 图表生成失败：{e}")
                            echarts_viz = {"show": False, "reason": str(e)}
                    else:
                        st.caption("📊 未安装 `streamlit-echarts`，暂不展示图表。")
                except Exception as e:
                    st.caption(f"⚠️ 查询结果展示失败：{e}")
            else:
                st.caption("⚠️ 未捕获到可用于展示的数据查询 SQL（last_sql 为空）。")

            # 显示 Token 使用统计（美化版）
            token_usage = tool_result.get("token_usage")
            if token_usage:
                input_tokens = token_usage.get("input_tokens", 0)
                output_tokens = token_usage.get("output_tokens", 0)
                total_tokens = token_usage.get("total_tokens", 0) or (input_tokens + output_tokens)
                llm_calls = token_usage.get("llm_calls", 0)
                if total_tokens > 0:
                    st.markdown("---")
                    st.markdown("##### 📊 Token 使用统计")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📥 输入", f"{input_tokens:,}")
                    with col2:
                        st.metric("📤 输出", f"{output_tokens:,}")
                    with col3:
                        st.metric("📊 总计", f"{total_tokens:,}")
                    with col4:
                        st.metric("🔄 调用次数", f"{llm_calls}")

            # 保存到消息历史：保存 last_sql + effective_limit + full_count + echarts_viz + trace
            msg = {"role": "assistant", "content": "", "question": prompt}
            if tool_result.get("sql"):
                msg["sql"] = tool_result["sql"]
            if "download_sql" in locals():
                msg["download_sql"] = download_sql
            if tool_result.get("last_sql"):
                msg["last_sql"] = tool_result["last_sql"]
            msg["effective_limit"] = int(tool_result.get("effective_limit") or 20)
            if full_count is not None:
                msg["full_count"] = int(full_count)
            if echarts_viz is not None:
                msg["echarts_viz"] = echarts_viz  # 缓存图表配置，历史渲染时直接使用
            if tool_result.get("trace"):
                msg["trace"] = tool_result["trace"]  # 保存工具调用轨迹
            if tool_result.get("token_usage"):
                msg["token_usage"] = tool_result["token_usage"]  # 保存 token 使用统计
            st.session_state.messages.append(msg)
            current_conv["updated_at"] = time.time()
        else:
            error_msg = f"❌ 查询失败: {tool_result.get('error', '未知错误')}"
            st.error(error_msg)
            # 失败时也显示已消耗的 token（简洁版）
            token_usage = tool_result.get("token_usage")
            if token_usage:
                total_tokens = token_usage.get("total_tokens", 0)
                llm_calls = token_usage.get("llm_calls", 0)
                if total_tokens > 0:
                    st.info(f"📊 已消耗 Token：**{total_tokens:,}**（{llm_calls} 次调用）")
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            current_conv["updated_at"] = time.time()

    # 清理运行标记并 rerun，恢复输入框
    st.session_state.pending_prompt = None
    st.session_state.is_running = False
    st.rerun()

# （已移除）示例问题、系统信息：保持聊天界面简洁
