"""
基于 Streamlit 的 SQL Agent Web 界面
提供聊天式交互和结果可视化
"""
import streamlit as st
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlagent import SQLAgent, Config

# 页面配置
st.set_page_config(
    page_title="SQL Agent - 智能查询助手",
    page_icon="🤖",
    layout="wide"
)

# 初始化 session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    try:
        # 显示加载提示
        with st.spinner("🔄 正在连接云端数据库并初始化Agent..."):
            st.session_state.agent = SQLAgent()
            st.session_state.db_name = Config.DB_NAME
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
                st.session_state.agent.switch_database(selected_db)
                st.session_state.db_name = selected_db
            st.success(f"已切换到: {selected_db}")
    except Exception as e:
        st.error(f"获取数据库列表失败: {e}")
    
    st.divider()
    
    # 显示当前数据库信息
    st.subheader("📊 数据库信息")
    if st.button("查看 Schema"):
        schema_info = st.session_state.agent.get_schema_info()
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
        st.markdown(message["content"])
        if "sql" in message:
            with st.expander("查看生成的 SQL"):
                st.code(message["sql"], language="sql")

# 输入框
if prompt := st.chat_input("请输入您的问题..."):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 显示加载状态
    with st.chat_message("assistant"):
        with st.spinner("正在查询..."):
            result = st.session_state.agent.query(prompt)
        
        if result["success"]:
            st.markdown(result["answer"])
            
            # 尝试提取并显示 SQL（如果可用）
            # 注意：这需要修改 agent.py 返回 SQL
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"]
            })
        else:
            error_msg = f"❌ 查询失败: {result.get('error', '未知错误')}"
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg
            })

# 示例问题
with st.expander("💡 示例问题"):
    st.markdown("""
    - 查询所有表的名称
    - 显示前10个客户的信息
    - 统计每个产品的销售数量
    - 查询最近一个月的交易记录
    - 显示数据库中有哪些表
    """)

# 显示系统信息
with st.expander("ℹ️ 系统信息"):
    st.json({
        "数据库": st.session_state.db_name,
        "模型": Config.MODEL_NAME,
        "最大迭代": Config.MAX_ITERATIONS,
        "默认限制": Config.DEFAULT_LIMIT
    })

