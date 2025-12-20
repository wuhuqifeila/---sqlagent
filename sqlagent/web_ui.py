"""
基于 Streamlit 的 SQL Agent Web 界面
提供聊天式交互和结果可视化
"""
import streamlit as st
import sys
import os
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import BaseCallbackHandler
import time

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlagent import SQLAgent, Config


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

            # 第2阶段：只流式输出“最终回答”
            answer_placeholder = st.empty()
            answer_stream_handler = StreamlitAnswerStreamHandler(answer_placeholder)
            final_answer = agent.stream_final_answer(
                question=prompt,
                sql=tool_result.get("sql", ""),
                sql_output=tool_result.get("sql_output", ""),
                callbacks=[answer_stream_handler],
            )
            # 确保页面上是完整文本
            streamed_text = answer_stream_handler.flush()
            final_answer = streamed_text or final_answer

            # 保存到消息历史（用于刷新后仍可见）
            message_data = {
                "role": "assistant",
                "content": final_answer
            }
            # 如果有SQL，也保存到消息历史中
            if tool_result.get("sql"):
                message_data["sql"] = tool_result["sql"]
            
            st.session_state.messages.append(message_data)
        else:
            error_msg = f"❌ 查询失败: {tool_result.get('error', '未知错误')}"
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg
            })

# （已移除）示例问题、系统信息：保持聊天界面简洁
