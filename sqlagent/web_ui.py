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

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlagent import SQLAgent, Config
from sqlagent.security import sanitize_sql_query, MAX_HARD_LIMIT

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
if "messages" not in st.session_state:
    st.session_state.messages = []

if "db_name" not in st.session_state:
    st.session_state.db_name = Config.DB_NAME

# 用于更顺滑的“运行中禁用输入框”体验（两段式 rerun）
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "is_running" not in st.session_state:
    st.session_state.is_running = False

# 获取缓存的 Agent（首次加载会显示加载提示）
try:
    with st.spinner("正在连接云端数据库并初始化Agent..."):
        agent = get_sql_agent(st.session_state.db_name)
except Exception as e:
    st.error(f"初始化 Agent 失败: {e}")
    st.stop()

# 侧边栏配置
with st.sidebar:
    st.title("⚙️ 配置")
    
    # 数据库选择（使用缓存避免重复查询）
    @st.cache_data(ttl=300)  # 缓存5分钟
    def get_databases():
        return Config.get_available_databases()
    
    try:
        databases = get_databases()
        selected_db = st.selectbox(
            "选择数据库",
            databases,
            index=databases.index(st.session_state.db_name) if st.session_state.db_name in databases else 0
        )
        
        if selected_db != st.session_state.db_name:
            with st.spinner(f"切换到数据库 {selected_db}..."):
                # 清除缓存，重新获取新数据库的 Agent
                get_sql_agent.clear()
                st.session_state.db_name = selected_db
                st.rerun()  # 重新运行以使用新的数据库
    except Exception as e:
        st.error(f"获取数据库列表失败: {e}")
    
    st.divider()
    
    # 显示当前数据库信息
    st.subheader("📊 数据库信息")
    if st.button("查看 Schema"):
        schema_info = agent.get_schema_info()
        st.write(f"**数据库**: {schema_info['database']}")
        st.write(f"**表列表**: {', '.join(schema_info['tables'])}")
        with st.expander("详细结构"):
            st.code(schema_info['table_info'], language="sql")
    
    st.divider()
    
    # 清空对话
    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.rerun()

# 主界面
st.title("🤖 SQL Agent - 智能 MySQL 查询助手")
st.caption(f"当前数据库: **{st.session_state.db_name}**")

# 显示对话历史（使用缓存的数据，避免重复渲染）
for msg_idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        # 如果是助手消息且有SQL，先显示SQL
        if message["role"] == "assistant" and "sql" in message:
            with st.expander("📝 查看生成的 SQL 语句", expanded=False):
                st.code(message["sql"], language="sql")

        # 如果历史消息里有 last_sql，则重绘"表格 + 下载 + 图表"
        if message["role"] == "assistant" and message.get("last_sql"):
            try:
                # 历史重绘也走统一 LIMIT 规则（≤20）
                eff = int(message.get("effective_limit") or 20)
                eff = max(1, min(eff, MAX_HARD_LIMIT))
                limited_sql = sanitize_sql_query(message["last_sql"], default_limit=eff, hard_limit=MAX_HARD_LIMIT)
                df_hist = get_df_for_sql(st.session_state.db_name, limited_sql)
                effective_limit = int(message.get("effective_limit") or 20)
                PREVIEW_ROWS = min(effective_limit, 20)
                preview_df = df_hist.head(PREVIEW_ROWS)
                
                # 显示全量行数（优先使用缓存的 full_count）
                full_count = message.get("full_count")
                if full_count is not None and full_count <= PREVIEW_ROWS:
                    st.markdown(f"**📄 查询结果（共 {full_count} 行，已全部展示）**")
                elif full_count is not None:
                    st.markdown(f"**📄 查询结果（前 {PREVIEW_ROWS} 行 / 共 {full_count} 行）**")
                elif len(df_hist) <= PREVIEW_ROWS:
                    st.markdown(f"**📄 查询结果（共 {len(df_hist)} 行，已全部展示）**")
                else:
                    st.markdown(f"**📄 查询结果（前 {PREVIEW_ROWS} 行 / 共 {len(df_hist)} 行）**")
                st.dataframe(preview_df, use_container_width=True)

                # 全量下载（Excel）— 需要唯一 key，避免 rerun 后组件状态错乱
                excel_bytes = build_excel_bytes(df_hist)
                filename = f"query_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                dl_key = f"download-{stable_key_for_sql(st.session_state.db_name, message['last_sql'])}"
                st.download_button(
                    label="⬇️ 下载全量结果（Excel）",
                    data=excel_bytes,
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=dl_key,
                )

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
                            df_info = build_df_info_for_viz(df_hist, max_rows=20)
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
                with st.expander("📝 查看生成的 SQL 语句", expanded=False):
                    st.code(tool_result["sql"], language="sql")

            last_sql = tool_result.get("last_sql", "") or ""
            if last_sql:
                try:
                    eff = int(tool_result.get("effective_limit") or 20)
                    eff = max(1, min(eff, MAX_HARD_LIMIT))
                    limited_sql = sanitize_sql_query(last_sql, default_limit=eff, hard_limit=MAX_HARD_LIMIT)

                    engine = get_sqlalchemy_engine(st.session_state.db_name)
                    df = execute_sql_to_df(limited_sql, engine)

                    # 计算"全量行数"：对去掉 LIMIT 的 SQL 做 COUNT(*)（失败则回退为 len(df)）
                    try:
                        sql_no_limit = strip_trailing_limit(last_sql)
                        count_sql = f"SELECT COUNT(*) FROM ({sql_no_limit}) AS t"
                        full_count = int(execute_scalar(count_sql, engine))
                    except Exception:
                        full_count = int(len(df)) if isinstance(df, pd.DataFrame) else None

                    PREVIEW_ROWS = min(eff, 20)
                    preview_df = df.head(PREVIEW_ROWS)
                    if full_count is not None and full_count <= PREVIEW_ROWS:
                        st.markdown(f"**📄 查询结果（共 {full_count} 行，已全部展示）**")
                    elif full_count is not None:
                        st.markdown(f"**📄 查询结果（前 {PREVIEW_ROWS} 行 / 共 {full_count} 行）**")
                    else:
                        st.markdown(f"**📄 查询结果（前 {PREVIEW_ROWS} 行）**")
                    st.dataframe(preview_df, use_container_width=True)

                    # 全量下载
                    excel_bytes = build_excel_bytes(df)
                    filename = f"query_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                    st.download_button(
                        label="⬇️ 下载全量结果（Excel）",
                        data=excel_bytes,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"download-live-{stable_key_for_sql(st.session_state.db_name, limited_sql)}",
                    )

                    # 可视化生成
                    if HAS_ECHARTS:
                        chart_placeholder = st.empty()
                        chart_placeholder.info("正在生成可视化图表...")
                        
                        try:
                            df_info = build_df_info_for_viz(df, max_rows=20)
                            viz = agent.generate_echarts_option(
                                question=prompt,
                                sql=tool_result.get("sql", ""),
                                df_info=df_info,
                            )
                            echarts_viz = viz  # 缓存到消息
                            
                            chart_placeholder.empty()  # 清除 loading 提示
                            if viz.get("show") and isinstance(viz.get("option"), dict):
                                st.markdown("**📊 可视化（ECharts）**")
                                st_echarts(viz["option"], height="420px", key=f"chart-live-{stable_key_for_sql(st.session_state.db_name, limited_sql)}")
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

            # 保存到消息历史：保存 last_sql + effective_limit + full_count + echarts_viz
            msg = {"role": "assistant", "content": ""}
            if tool_result.get("sql"):
                msg["sql"] = tool_result["sql"]
            if tool_result.get("last_sql"):
                msg["last_sql"] = tool_result["last_sql"]
            msg["effective_limit"] = int(tool_result.get("effective_limit") or 20)
            if full_count is not None:
                msg["full_count"] = int(full_count)
            if echarts_viz is not None:
                msg["echarts_viz"] = echarts_viz  # 缓存图表配置，历史渲染时直接使用
            st.session_state.messages.append(msg)
        else:
            error_msg = f"❌ 查询失败: {tool_result.get('error', '未知错误')}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # 清理运行标记并 rerun，恢复输入框
    st.session_state.pending_prompt = None
    st.session_state.is_running = False
    st.rerun()

# （已移除）示例问题、系统信息：保持聊天界面简洁
