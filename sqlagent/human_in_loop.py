"""
人工介入模块 (Human-in-the-loop)
使用 LangGraph 实现 SQL 执行前的审批流程
"""
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage

from .agent import SQLAgent
from .security import sanitize_sql_query


class AgentState(TypedDict):
    """Agent 状态定义"""
    messages: Annotated[list, lambda x, y: x + y]
    sql_query: str
    approval_status: Literal["pending", "approved", "rejected", "modified"]
    human_feedback: str


class HumanInLoopAgent:
    """带人工审批的 SQL Agent"""
    
    def __init__(self, sql_agent: SQLAgent):
        """
        初始化人工介入 Agent
        
        Args:
            sql_agent: SQL Agent 实例
        """
        self.sql_agent = sql_agent
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 工作流"""
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("generate_sql", self._generate_sql)
        workflow.add_node("wait_for_approval", self._wait_for_approval)
        workflow.add_node("execute_sql", self._execute_sql)
        workflow.add_node("format_response", self._format_response)
        
        # 定义边
        workflow.set_entry_point("generate_sql")
        workflow.add_edge("generate_sql", "wait_for_approval")
        workflow.add_conditional_edges(
            "wait_for_approval",
            self._check_approval,
            {
                "approved": "execute_sql",
                "rejected": END,
                "modified": "generate_sql"
            }
        )
        workflow.add_edge("execute_sql", "format_response")
        workflow.add_edge("format_response", END)
        
        # 添加检查点（用于中断和恢复）
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    def _generate_sql(self, state: AgentState) -> AgentState:
        """生成 SQL 查询"""
        question = state["messages"][-1].content if state["messages"] else ""
        
        # 使用 SQL Agent 生成 SQL（这里简化处理，实际应该提取 SQL）
        result = self.sql_agent.agent_executor.invoke({"input": question})
        
        # 从结果中提取 SQL（这里需要根据实际返回格式调整）
        sql_query = self._extract_sql_from_result(result)
        
        state["sql_query"] = sql_query
        state["approval_status"] = "pending"
        state["messages"].append(AIMessage(content=f"生成的 SQL 查询：\n```sql\n{sql_query}\n```\n\n请审批：输入 'approve' 执行，'reject' 拒绝，或提供修改意见。"))
        
        return state
    
    def _wait_for_approval(self, state: AgentState) -> AgentState:
        """等待人工审批（中断点）"""
        # 这里会中断，等待人工输入
        # 实际使用时需要通过 LangGraph 的 interrupt 机制
        return state
    
    def _check_approval(self, state: AgentState) -> str:
        """检查审批状态"""
        if not state.get("human_feedback"):
            return "pending"
        
        feedback = state["human_feedback"].lower().strip()
        
        if feedback == "approve" or feedback == "批准":
            state["approval_status"] = "approved"
            return "approved"
        elif feedback == "reject" or feedback == "拒绝":
            state["approval_status"] = "rejected"
            return "rejected"
        else:
            state["approval_status"] = "modified"
            return "modified"
    
    def _execute_sql(self, state: AgentState) -> AgentState:
        """执行 SQL 查询"""
        sql_query = state["sql_query"]
        
        try:
            # 安全过滤
            safe_sql = sanitize_sql_query(sql_query, self.sql_agent.default_limit)
            
            # 执行查询
            result = self.sql_agent.db.run(safe_sql)
            
            state["messages"].append(AIMessage(content=f"查询结果：\n{result}"))
        except Exception as e:
            state["messages"].append(AIMessage(content=f"执行失败：{str(e)}"))
        
        return state
    
    def _format_response(self, state: AgentState) -> AgentState:
        """格式化响应"""
        # 格式化最终响应
        return state
    
    def _extract_sql_from_result(self, result: dict) -> str:
        """从 Agent 结果中提取 SQL 语句"""
        # 这里需要根据实际返回格式提取 SQL
        # 简化实现
        if "intermediate_steps" in result:
            for step in result["intermediate_steps"]:
                if isinstance(step, tuple) and len(step) > 0:
                    tool_input = step[0]
                    if isinstance(tool_input, dict) and "query" in tool_input:
                        return tool_input["query"]
        return ""
    
    def query_with_approval(self, question: str, approval_callback=None):
        """
        执行需要审批的查询
        
        Args:
            question: 用户问题
            approval_callback: 审批回调函数，接收 SQL 查询，返回审批结果
        """
        config = {"configurable": {"thread_id": "1"}}
        
        # 初始化状态
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "sql_query": "",
            "approval_status": "pending",
            "human_feedback": ""
        }
        
        # 运行到审批节点
        result = self.graph.invoke(initial_state, config)
        
        # 如果有审批回调，调用它
        if approval_callback:
            sql_query = result["sql_query"]
            approval_result = approval_callback(sql_query)
            
            if approval_result == "approve":
                result["human_feedback"] = "approve"
            elif approval_result == "reject":
                result["human_feedback"] = "reject"
            else:
                result["human_feedback"] = approval_result
        
        # 继续执行
        final_result = self.graph.invoke(result, config)
        
        return final_result

