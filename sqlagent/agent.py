"""
SQL Agent 核心模块
参考 LangChain 官方文档简化实现
"""
import os
import json
import ast
import re
from typing import Optional, Dict, Any, Callable, List, Optional as TypingOptional
from langchain_community.utilities import SQLDatabase
from langchain_community.chat_models import ChatTongyi
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.callbacks import BaseCallbackHandler

from .config import Config
from .security import sanitize_sql_query


MAX_HARD_LIMIT = 20


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


def extract_requested_limit(question: str, hard_cap: int = MAX_HARD_LIMIT) -> int:
    """
    从用户问题中抽取希望返回的行数 N（若未指定则默认 hard_cap），并强制不超过 hard_cap。
    支持：前10/Top10/只要5条/限制20行/返回十条 等。
    """
    q = (question or "").strip()
    if not q:
        return hard_cap

    # 先找阿拉伯数字
    m = re.search(r"(?i)(?:top|前|最多|只要|限制|返回|显示|取)\s*(\d{1,4})\s*(?:条|行)?", q)
    if m:
        n = int(m.group(1))
        return max(1, min(n, hard_cap))

    # 再找中文数字（十/二十/二十一等）
    m2 = re.search(r"(?:top|前|最多|只要|限制|返回|显示|取)\s*([一二两三四五六七八九十]{1,3})\s*(?:条|行)?", q)
    if m2:
        n2 = _chinese_num_to_int(m2.group(1))
        if n2 is not None:
            return max(1, min(n2, hard_cap))

    return hard_cap


class AgentTraceHandler(BaseCallbackHandler):
    """捕获 Agent 工具调用过程（用于前端展示）并提取 SQL。"""

    def __init__(self) -> None:
        self.sql_queries: List[str] = []
        self.trace: List[Dict[str, Any]] = []
        self._run_id_to_index: Dict[str, int] = {}

    @staticmethod
    def _normalize_input(value: Any) -> str:
        """将工具输入尽量规范化为可展示的字符串。"""
        if value is None:
            return ""
        if isinstance(value, str):
            s = value.strip()
            # 兼容某些版本里把 {"query": "..."} 以字符串形式传入的情况
            if s.startswith("{") and "query" in s:
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, dict) and isinstance(parsed.get("query"), str):
                        return parsed["query"].strip()
                except Exception:
                    # 再兜底一次：用正则提取 query
                    m = re.search(r"""['"]query['"]\s*:\s*['"]([\s\S]*?)['"]\s*\}?$""", s)
                    if m:
                        return m.group(1).strip()
            return s
        if isinstance(value, dict):
            # 常见字段：query / tool_input / table_names
            if "query" in value and isinstance(value["query"], str):
                return value["query"].strip()
            if "tool_input" in value and isinstance(value["tool_input"], str):
                return value["tool_input"].strip()
            try:
                return json.dumps(value, ensure_ascii=False, indent=2)
            except Exception:
                return str(value)
        return str(value).strip()

    def on_tool_start(self, serialized: Dict[str, Any], input_str: Any = None, **kwargs) -> None:
        """工具开始执行时触发。"""
        tool_name = (serialized or {}).get("name") or kwargs.get("name") or ""
        run_id = str(kwargs.get("run_id") or "")

        normalized_input = self._normalize_input(input_str if input_str is not None else kwargs.get("input"))

        # 记录 trace
        entry: Dict[str, Any] = {
            "tool": tool_name,
            "input": normalized_input,
            "output": None,
        }
        self.trace.append(entry)
        if run_id:
            self._run_id_to_index[run_id] = len(self.trace) - 1

        # 捕获 SQL
        if tool_name == "sql_db_query":
            if normalized_input:
                self.sql_queries.append(normalized_input)

    def on_tool_end(self, output: Any, **kwargs) -> None:
        """工具执行结束时触发。"""
        run_id = str(kwargs.get("run_id") or "")
        normalized_output = self._normalize_input(output)

        idx: TypingOptional[int] = self._run_id_to_index.get(run_id) if run_id else None
        if idx is None:
            # 找不到 run_id 时，兜底：给最后一个尚未填充 output 的条目
            for i in range(len(self.trace) - 1, -1, -1):
                if self.trace[i].get("output") is None:
                    idx = i
                    break
        if idx is not None:
            self.trace[idx]["output"] = normalized_output

    def on_tool_error(self, error: Exception, **kwargs) -> None:
        run_id = str(kwargs.get("run_id") or "")
        idx = self._run_id_to_index.get(run_id) if run_id else None
        if idx is None and self.trace:
            idx = len(self.trace) - 1
        if idx is not None:
            self.trace[idx]["output"] = f"Error: {error}"


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
        return f"""你是一名严谨的 MySQL 数据分析助手。

【权威 Schema（严禁臆造表/字段）】
{schema_info}

【规则】
- 你需要查询数据时，只能调用工具 `sql_db_query`，并且一次只提交 **一条 SELECT** 查询。
- 只读：禁止 INSERT / UPDATE / DELETE / ALTER / DROP / CREATE / REPLACE / TRUNCATE 等写操作。
- 除非用户明确要求更多，否则默认最多返回 {default_limit} 行。
- 如果工具返回 Error，请修正 SQL 后再重试。
- 最多尝试 10 次；若仍失败，向用户说明原因并给出建议。
- 尽量写明列名，避免 SELECT *。
- 最终回答必须使用中文。
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
        default_limit: int = Config.DEFAULT_LIMIT,
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
        
        # 配置连接池参数，提升连接性能和稳定性
        engine_args = {
            "pool_pre_ping": True,      # 使用前检测连接，自动处理断线
            "pool_size": 5,              # 连接池大小
            "max_overflow": 10,          # 最大溢出连接数
            "pool_recycle": 3600,        # 1小时回收连接
            "connect_args": {
                "connect_timeout": 10    # 连接超时10秒
            }
        }
        
        self.db = SQLDatabase.from_uri(db_uri, engine_args=engine_args)
        self.db_name = target_db
        self.default_limit = default_limit
        # 当前问题的有效 LIMIT（每次 query/run_tools 时更新），用于强制钳制 sql_db_query 的返回行数
        self._effective_limit: int = min(int(default_limit), MAX_HARD_LIMIT)

        # Monkey-patch：强制限制 sql_db_query 实际执行的 SQL 行数（不依赖提示词）
        self._patch_db_limit()
        
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

    def _patch_db_limit(self) -> None:
        """拦截 SQLDatabase.run / run_no_throw，在执行前强制追加/收缩 LIMIT（≤20）。"""
        if not hasattr(self, "db") or self.db is None:
            return

        db = self.db

        def _wrap_run(fn):
            # fn 是“已绑定方法”（bound method），直接调用即可；不要再用 MethodType 二次绑定
            def _inner(command: str, *args, **kwargs):
                try:
                    # 只对 SELECT 做限流；sanitize_sql_query 内部也会拒绝非 SELECT
                    limited = sanitize_sql_query(command, default_limit=self._effective_limit)
                except Exception:
                    # 若 sanitize 失败，仍执行原 SQL（避免影响 list/schema 等非 SELECT 场景）
                    limited = command
                return fn(limited, *args, **kwargs)
            return _inner

        if hasattr(db, "run") and callable(getattr(db, "run")):
            original_run = db.run  # bound method
            db.run = _wrap_run(original_run)  # type: ignore
        if hasattr(db, "run_no_throw") and callable(getattr(db, "run_no_throw")):
            original_run_no_throw = db.run_no_throw  # bound method
            db.run_no_throw = _wrap_run(original_run_no_throw)  # type: ignore
    
    def query(self, question: str, callbacks: Optional[List[BaseCallbackHandler]] = None) -> Dict[str, Any]:
        """
        执行自然语言查询
        
        Args:
            question: 用户问题（作为 user prompt 传入）
        
        Returns:
            查询结果字典，包含：
            - success: 是否成功
            - question: 用户问题
            - answer: AI回答
            - sql: 生成的SQL语句（如果有）
            - database: 数据库名称
        
        注意：
            - System Prompt: 由 create_sql_agent 内部自动生成，包含数据库 Schema 和规则
            - User Prompt: 通过 {"input": question} 传入，即用户的问题
        """
        # 创建 Trace 捕获处理器（工具调用过程 + SQL）
        trace_handler = AgentTraceHandler()
        extra_callbacks = callbacks or []
        callbacks_list: List[BaseCallbackHandler] = [trace_handler, *extra_callbacks]

        # 每轮根据用户输入计算有效 LIMIT（≤20），用于强制钳制 sql_db_query 返回行数
        self._effective_limit = extract_requested_limit(question, hard_cap=MAX_HARD_LIMIT)
        
        try:
            # 使用回调处理器来捕获工具调用过程与SQL
            result = self.agent_executor.invoke(
                {"input": question},
                config={"callbacks": callbacks_list}
            )
            
            response = {
                "success": True,
                "question": question,
                "answer": result.get("output", ""),
                "database": self.db_name
            }
            
            # 如果捕获到工具调用过程，返回给前端展示（可折叠）
            if trace_handler.trace:
                response["trace"] = trace_handler.trace

            # 如果捕获到SQL，添加到返回结果中（用于专门的SQL下拉框）
            if trace_handler.sql_queries:
                response["sql"] = "\n\n".join(trace_handler.sql_queries)
            
            return response
            
        except Exception as e:
            return {
                "success": False,
                "question": question,
                "error": str(e),
                "database": self.db_name
            }

    def run_tools(self, question: str, callbacks: Optional[List[BaseCallbackHandler]] = None) -> Dict[str, Any]:
        """
        只运行“工具调用/SQL执行”阶段，返回 SQL、工具 trace、以及最后一次 SQL 查询结果（字符串）。
        用于：先拿到 SQL 再开始流式生成最终回答。
        """
        trace_handler = AgentTraceHandler()
        extra_callbacks = callbacks or []
        callbacks_list: List[BaseCallbackHandler] = [trace_handler, *extra_callbacks]

        # 每轮根据用户输入计算有效 LIMIT（≤20），用于强制钳制 sql_db_query 返回行数
        self._effective_limit = extract_requested_limit(question, hard_cap=MAX_HARD_LIMIT)

        try:
            result = self.agent_executor.invoke(
                {"input": question},
                config={"callbacks": callbacks_list},
            )

            sql_text = "\n\n".join(trace_handler.sql_queries) if trace_handler.sql_queries else ""
            last_sql = trace_handler.sql_queries[-1].strip() if trace_handler.sql_queries else ""

            # 找到最后一次 sql_db_query 的输出（作为最终回答的依据）
            last_sql_output = ""
            for step in reversed(trace_handler.trace):
                if step.get("tool") == "sql_db_query" and step.get("output"):
                    last_sql_output = step.get("output", "")
                    break

            return {
                "success": True,
                "question": question,
                "database": self.db_name,
                "effective_limit": self._effective_limit,
                "trace": trace_handler.trace,
                "sql": sql_text,
                "last_sql": last_sql,
                "sql_output": last_sql_output,
                # 保留 agent 原始 output 以便兜底
                "agent_output": result.get("output", ""),
            }
        except Exception as e:
            return {
                "success": False,
                "question": question,
                "database": self.db_name,
                "error": str(e),
            }

    def stream_final_answer(
        self,
        question: str,
        sql: str,
        sql_output: str,
        *,
        total_rows: Optional[int] = None,
        preview_rows_text: Optional[str] = None,
        max_preview_rows: int = 20,
        callbacks: Optional[List[BaseCallbackHandler]] = None,
    ) -> str:
        """
        仅流式生成“最终回答”。工具阶段已完成后调用本方法。
        """
        final_llm = ChatTongyi(
            model_name=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            streaming=True,
            callbacks=callbacks or [],
        )

        system = SystemMessage(
            content=(
                "你是财务数据分析助手。请根据用户问题、SQL 以及查询结果信息，用中文给出清晰、准确的最终回答。\n"
                "要求：不要输出推理过程，不要输出工具调用痕迹。\n"
                "【强制约束（必须遵守）】\n"
                "- 最终回答**只输出一段中文总结**（允许分句，但不要用列表/表格/逐行罗列）。\n"
                "- 不得逐条列出记录明细，不得输出表格，不得输出逐行列表。\n"
                "- 即使用户要求“全部”，也只允许做汇总性的说明，并引导用户到页面下方表格/下载查看全量结果。\n"
            )
        )
        human = HumanMessage(
            content=(
                f"【用户问题】\n{question}\n\n"
                f"【生成的SQL】\n{sql or '(无)'}\n\n"
                f"【查询结果总行数】\n{total_rows if total_rows is not None else '(未知)'}\n\n"
                f"【结果样例（最多 {max_preview_rows} 行，仅供你总结用，不要逐条复述）】\n{preview_rows_text or '(无样例)'}\n\n"
                f"【原始SQL返回结果（可能已截断/不完整）】\n{sql_output or '(无)'}\n\n"
                "请给出最终回答（只写一段总结，不要列数据，不要列表）："
            )
        )
        msg = final_llm.invoke([system, human])
        # Chat message -> content
        return getattr(msg, "content", str(msg))

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        """从模型输出中尽量提取 JSON 对象。"""
        if not text:
            return None
        t = text.strip()
        # 去掉代码围栏
        if t.startswith("```"):
            t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
            t = re.sub(r"\s*```$", "", t).strip()
        # 尝试直接解析
        try:
            obj = json.loads(t)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        # 尝试提取第一个 {...}
        m = re.search(r"\{[\s\S]*\}", t)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict):
                    return obj
            except Exception:
                return None
        return None

    def generate_echarts_option(
        self,
        *,
        question: str,
        sql: str,
        df_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        让 LLM 根据 SQL 结果的摘要信息生成 ECharts option。
        返回格式（JSON dict）：
        - show: bool
        - reason: str（可选）
        - option: dict（show=true 时必需）
        """
        viz_llm = ChatTongyi(
            model_name=Config.MODEL_NAME,
            temperature=0,
            streaming=False,
        )

        system = SystemMessage(
            content=(
                "你是数据可视化助手。你要根据给定的 SQL 结果摘要，自动选择合适的 ECharts 图表并输出 ECharts option。\n"
                "必须严格输出一个 JSON 对象（不要 Markdown，不要解释性文字）。\n"
                "如果数据不适合绘图（例如：全是ID/编码/高基数字符串、或无数值/无可聚合分类、或结果为空），请输出：\n"
                "{\"show\": false, \"reason\": \"原因\"}\n"
                "如果适合绘图，请输出：\n"
                "{\"show\": true, \"option\": { ...ECharts option... }}\n"
                "要求：option 尽量简洁；若类别过多只取 Top 20；轴标题使用中文；tooltip 开启。\n"
                "禁止使用 dataset/transform 等高级特性（保证兼容性）；直接在 series.data 里给数组。\n"
            )
        )

        human = HumanMessage(
            content=(
                f"用户问题：{question}\n"
                f"SQL：{sql}\n"
                f"结果摘要(JSON)：{json.dumps(df_info, ensure_ascii=False)}\n"
            )
        )

        msg = viz_llm.invoke([system, human])
        text = getattr(msg, "content", str(msg))
        obj = self._extract_json(text)
        if not obj:
            return {"show": False, "reason": "模型未返回可解析的 JSON"}

        # 规范化
        show = bool(obj.get("show", False))
        if show and not isinstance(obj.get("option"), dict):
            return {"show": False, "reason": "模型返回 show=true 但缺少 option"}
        return obj
    
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

