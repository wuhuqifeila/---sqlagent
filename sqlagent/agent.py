"""
SQL Agent 核心模块
参考 LangChain 官方文档简化实现
"""
import os
from typing import Optional, Dict, Any, Callable
from langchain_community.utilities import SQLDatabase
from langchain_community.chat_models import ChatTongyi
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.messages import SystemMessage

from .config import Config


class SQLAgent:
    """SQL Agent 主类"""
    
    @staticmethod
    def _create_default_system_prompt(schema_info: str, default_limit: int) -> str:
        """
        创建默认的 System Prompt
        
        Args:
            schema_info: 数据库 Schema 信息
            default_limit: 默认查询限制行数
        
        Returns:
            System Prompt 字符串
        """
        return f"""You are a careful MySQL analyst.

Authoritative schema (do not invent columns/tables):

{schema_info}

Rules:

- Think step-by-step.

- When you need data, call the tool `sql_db_query` with ONE SELECT query.

- Read-only only; no INSERT/UPDATE/DELETE/ALTER/DROP/CREATE/REPLACE/TRUNCATE.

- Limit to {default_limit} rows unless user explicitly asks otherwise.

- If the tool returns 'Error:', revise the SQL and try again.

- Limit the number of attempts to 5.

- If you are not successful after 5 attempts, return a note to the user.

- Prefer explicit column lists; avoid SELECT *.

- Response in Chinese.
"""
    """SQL Agent 主类"""
    
    def __init__(
        self,
        db_name: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_iterations: Optional[int] = None,
        verbose: Optional[bool] = None,
        agent_type: Optional[str] = None,
        default_limit: int = 5,
        system_prompt: Optional[str] = None
    ):
        """
        初始化 SQL Agent
        
        Args:
            db_name: 数据库名称
            model_name: 模型名称
            temperature: 温度参数
            max_iterations: 最大迭代次数
            verbose: 是否显示详细信息
            agent_type: Agent 类型
            default_limit: 默认查询限制行数
            system_prompt: 自定义系统提示词（如果为 None，则使用默认提示词）
        """
        # 设置环境变量
        os.environ["DASHSCOPE_API_KEY"] = Config.DASHSCOPE_API_KEY
        
        # 数据库连接
        target_db = db_name or Config.DB_NAME
        if target_db is None:
            raise ValueError("数据库名称未指定，请设置 DB_NAME 环境变量或传入 db_name 参数")
        
        db_uri = Config.get_db_uri(target_db)
        self.db = SQLDatabase.from_uri(db_uri)
        self.db_name = target_db
        self.default_limit = default_limit
        
        # LLM 初始化
        self.llm = ChatTongyi(
            model_name=model_name or Config.MODEL_NAME,
            temperature=temperature if temperature is not None else Config.TEMPERATURE
        )
        
        # 创建自定义 System Prompt（参考文档）
        if system_prompt is None:
            # 使用默认 System Prompt
            schema_info = self.db.get_table_info()
            system_prompt = self._create_default_system_prompt(schema_info, self.default_limit)
        
        # 创建 SystemMessage
        system_message = SystemMessage(content=system_prompt)
        
        # 创建 Agent（尝试传入自定义 prompt）
        agent_kwargs = {
            "llm": self.llm,
            "db": self.db,
            "verbose": verbose if verbose is not None else Config.VERBOSE,
            "max_iterations": max_iterations or Config.MAX_ITERATIONS,
            "agent_type": "openai-tools" if (agent_type or Config.AGENT_TYPE) == "openai-tools" else "zero-shot-react-description"
        }
        
        # 尝试添加自定义 system prompt（如果 create_sql_agent 支持）
        try:
            agent_kwargs["system_message"] = system_message
        except TypeError:
            # 如果不支持 system_message 参数，尝试其他方式
            try:
                agent_kwargs["prompt"] = system_message
            except TypeError:
                # 如果都不支持，使用默认方式（create_sql_agent 会自动生成 prompt）
                pass
        
        self.agent_executor = create_sql_agent(**agent_kwargs)
        
        # 保存 system prompt 供查看
        self.system_prompt = system_prompt
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        执行自然语言查询
        
        Args:
            question: 用户问题（作为 user prompt 传入）
        
        Returns:
            查询结果字典
        
        注意：
            - System Prompt: 由 create_sql_agent 内部自动生成，包含数据库 Schema 和规则
            - User Prompt: 通过 {"input": question} 传入，即用户的问题
        """
        try:
            # User Prompt 在这里：{"input": question}
            result = self.agent_executor.invoke({"input": question})
            return {
                "success": True,
                "question": question,
                "answer": result.get("output", ""),
                "database": self.db_name
            }
        except Exception as e:
            return {
                "success": False,
                "question": question,
                "error": str(e),
                "database": self.db_name
            }
    
    def get_schema_info(self) -> Dict[str, Any]:
        """
        获取数据库 Schema 信息
        
        Returns:
            Schema 信息字典
        
        注意：这个 Schema 信息会被自动包含在 System Prompt 中
        """
        return {
            "database": self.db_name,
            "tables": self.db.get_usable_table_names(),
            "table_info": self.db.get_table_info()
        }
    
    def get_prompt_info(self) -> Dict[str, Any]:
        """
        获取 Agent 使用的 Prompt 信息
        
        Returns:
            Prompt 信息字典
        """
        return {
            "system_prompt": self.system_prompt,
            "schema_info": self.db.get_table_info()[:500] + "..." if len(self.db.get_table_info()) > 500 else self.db.get_table_info(),
            "user_prompt_format": '{"input": "用户问题"}',
            "default_limit": self.default_limit
        }
    
    def switch_database(self, db_name: str):
        """切换数据库"""
        db_uri = Config.get_db_uri(db_name)
        self.db = SQLDatabase.from_uri(db_uri)
        self.db_name = db_name
        
        # 更新 system prompt 中的 schema 信息
        schema_info = self.db.get_table_info()
        self.system_prompt = self._create_default_system_prompt(schema_info, self.default_limit)
        
        # 重新创建 Agent
        system_message = SystemMessage(content=self.system_prompt)
        agent_kwargs = {
            "llm": self.llm,
            "db": self.db,
            "verbose": Config.VERBOSE,
            "max_iterations": Config.MAX_ITERATIONS,
            "agent_type": "openai-tools" if Config.AGENT_TYPE == "openai-tools" else "zero-shot-react-description"
        }
        
        try:
            agent_kwargs["system_message"] = system_message
        except TypeError:
            try:
                agent_kwargs["prompt"] = system_message
            except TypeError:
                pass
        
        self.agent_executor = create_sql_agent(**agent_kwargs)

