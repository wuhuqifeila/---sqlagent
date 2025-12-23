"""
基于 Streamlit 的 SQL Agent Web 界面
提供聊天式交互和结果可视化
"""
import streamlit as st
import sys
import os
from io import BytesIO
from datetime import datetime
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import BaseCallbackHandler
import time
import pandas as pd
from sqlalchemy import create_engine
import hashlib

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlagent import SQLAgent, Config

# ECharts（可选依赖，未安装时自动降级不显示图表）
try:
    from streamlit_echarts import st_echarts  # type: ignore
    HAS_ECHARTS = True
except Exception:
    HAS_ECHARTS = False


class StreamlitStatusTraceHandler(BaseCallbackHandler):
    """把工具调用过程实时写到 Streamlit 的 st.status（默认折叠 + 运行状态）。"""

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

# 获取缓存的 Agent（首次加载会显示加载提示）
try:
    with st.spinner("🔄 正在连接云端数据库并初始化Agent..."):
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

# 显示对话历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # 如果是助手消息且有SQL，先显示SQL
        if message["role"] == "assistant" and "sql" in message:
            with st.expander("📝 查看生成的 SQL 语句", expanded=False):
                st.code(message["sql"], language="sql")

        # 如果历史消息里有 last_sql，则重绘“表格 + 下载 + 图表”
        if message["role"] == "assistant" and message.get("last_sql"):
            try:
                df_hist = get_df_for_sql(st.session_state.db_name, message["last_sql"])
                effective_limit = int(message.get("effective_limit") or 20)
                PREVIEW_ROWS = min(effective_limit, 20)
                preview_df = df_hist.head(PREVIEW_ROWS)
                if len(df_hist) <= PREVIEW_ROWS:
                    st.markdown(f"**📄 查询结果（共 {len(df_hist)} 行，已全部展示）**")
                else:
                    st.markdown(f"**📄 查询结果（前 {PREVIEW_ROWS} 行 / 共 {len(df_hist)} 行）**")
                st.dataframe(preview_df, width="stretch")

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

                if HAS_ECHARTS:
                    option = auto_echarts_option(df_hist)
                    if option:
                        st.markdown("**📊 可视化（ECharts）**")
                        st_echarts(option, height="420px", key=f"chart-{dl_key}")
            except Exception as e:
                st.caption(f"⚠️ 查询结果展示失败：{e}")
        
        # 显示消息内容
        st.markdown(message["content"])

# 输入框
if prompt := st.chat_input("请输入您的问题..."):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 显示加载状态
    with st.chat_message("assistant"):
        # 默认折叠，但会显示“运行中”的状态块（用户可点开看过程）
        status_box = st.status("正在查询数据库…", expanded=False, state="running")
        live_handler = StreamlitStatusTraceHandler(status_box)

        # 第1阶段：运行工具/SQL（不流式）
        tool_result = agent.run_tools(prompt, callbacks=[live_handler])

        # 查询结束：更新状态（仍然保持折叠，避免占页面）
        if tool_result.get("success"):
            status_box.update(label="数据库查询完成", state="complete", expanded=False)
        else:
            status_box.update(label="数据库查询失败", state="error", expanded=False)

        if tool_result.get("success"):
            # 如果有生成的SQL，先在可折叠框中展示（在最终回答之前）
            if tool_result.get("sql"):
                with st.expander("📝 查看生成的 SQL 语句", expanded=False):
                    st.code(tool_result["sql"], language="sql")

            # SQL 下方展示数据（前10行）+ 全量下载 + ECharts
            last_sql = tool_result.get("last_sql", "") or ""
            if last_sql:
                try:
                    engine = get_sqlalchemy_engine(st.session_state.db_name)
                    df = execute_sql_to_df(last_sql, engine)

                    effective_limit = int(tool_result.get("effective_limit") or 20)
                    PREVIEW_ROWS = min(effective_limit, 20)
                    preview_df = df.head(PREVIEW_ROWS)
                    if len(df) <= PREVIEW_ROWS:
                        st.markdown(f"**📄 查询结果（共 {len(df)} 行，已全部展示）**")
                    else:
                        st.markdown(f"**📄 查询结果（前 {PREVIEW_ROWS} 行 / 共 {len(df)} 行）**")
                    # Streamlit 新版推荐用 width="stretch"
                    st.dataframe(preview_df, width="stretch")

                    # 全量下载（Excel）
                    excel_bytes = build_excel_bytes(df)
                    filename = f"query_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                    st.download_button(
                        label="⬇️ 下载全量结果（Excel）",
                        data=excel_bytes,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

                    # ECharts 可视化
                    if HAS_ECHARTS:
                        option = auto_echarts_option(df)
                        if option:
                            st.markdown("**📊 可视化（ECharts）**")
                            st_echarts(option, height="420px")
                        else:
                            st.caption("📊 当前结果不适合自动绘图（列类型不足或数据为空）。")
                    else:
                        st.caption("📊 未安装 `streamlit-echarts`，暂不展示图表。")
                except Exception as e:
                    st.caption(f"⚠️ 查询结果展示失败：{e}")
            else:
                st.caption("⚠️ 未捕获到可用于展示的数据查询 SQL（last_sql 为空）。")

            # 不再调用大模型生成“最终总结回复”，改为固定模板（零额外 token）
            total_rows = int(df.shape[0]) if "df" in locals() and isinstance(df, pd.DataFrame) else None
            if total_rows is not None:
                final_answer = f"查询完成，共 {total_rows} 行结果。明细请查看上方表格，或点击下方按钮下载全量 Excel。"
            else:
                final_answer = "查询完成。明细请查看上方表格，或点击下方按钮下载全量 Excel。"
            st.markdown(final_answer)

            # 保存到消息历史（用于刷新后仍可见）
            message_data = {
                "role": "assistant",
                "content": final_answer
            }
            # 如果有SQL，也保存到消息历史中
            if tool_result.get("sql"):
                message_data["sql"] = tool_result["sql"]
            # 保存 last_sql，用于 rerun 后重绘表格/下载/图表
            if tool_result.get("last_sql"):
                message_data["last_sql"] = tool_result["last_sql"]
            message_data["effective_limit"] = int(tool_result.get("effective_limit") or 20)
            
            st.session_state.messages.append(message_data)
        else:
            error_msg = f"❌ 查询失败: {tool_result.get('error', '未知错误')}"
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg
            })

# （已移除）示例问题、系统信息：保持聊天界面简洁
