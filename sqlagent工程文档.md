工程文档：基于 LangChain + Qwen 的智能 MySQL 查询 Agent
1. 项目概述
本项目旨在构建一个基于 LangChain 框架和 Qwen 大模型的智能数据库助手。该 Agent 能够通过自然语言理解用户意图，自动生成、校验并执行 SQL 语句，最终从 MySQL 数据库中提取数据并回馈用户。

2. 核心工作流
Agent 的运行遵循以下逻辑步骤（基于 ReAct 模式）：

架构感知：获取数据库表名及结构（Schema）。

查询规划：确定与问题相关的表。

SQL 生成：根据用户问题和 Schema 生成 SQL 语句。

安全审查与自检：通过 LLM 检查 SQL 语法及安全风险，并在执行前注入 LIMIT 限制。

纠错循环 (Self-Correction)：若执行报错，Agent 将获取错误信息并自动重试修复（最多 5 次）。

结果转换：将结构化数据转化为自然语言回复。

3. 技术栈
LLM: Qwen (通过 ChatTongyi 调用)

框架: LangChain / LangGraph

数据库: MySQL 8.0+

连接驱动: PyMySQL, SQLAlchemy

核心库: langchain-community, langchain-core

4. 核心模块实现
4.1 环境准备
Bash

pip install langchain langchain-community langchain-openai pymysql sqlalchemy dashscope langgraph
4.2 SQL 安全过滤 (Sanitization)
为防止恶意 SQL 注入及意外删除，我们在执行前定义过滤逻辑。

Python

import re

def sanitize_sql_query(query: str) -> str:
    # 禁止 DML/DDL 关键字
    deny_pattern = r"\b(INSERT|UPDATE|DELETE|ALTER|DROP|CREATE|REPLACE|TRUNCATE)\b"
    if re.search(deny_pattern, query, re.IGNORECASE):
        raise ValueError("检测到非查询语句，仅允许 SELECT 操作。")
    
    # 强制增加 LIMIT 5 保护
    if "limit" not in query.lower():
        query = query.rstrip(';') + " LIMIT 5;"
    return query
4.3 工具与 Agent 初始化
集成 Qwen 模型并配置具有“自愈”功能的 Agent。

Python

from langchain_community.utilities import SQLDatabase
from langchain_community.chat_models import ChatTongyi
from langchain_community.agent_toolkits import create_sql_agent

# 1. 数据库连接
db = SQLDatabase.from_uri("mysql+pymysql://user:pass@localhost/db_name")

# 2. 模型初始化 (Qwen-Max)
llm = ChatTongyi(model_name="qwen-max", temperature=0)

# 3. 创建 Agent (内置了自动重试逻辑)
agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    verbose=True,
    max_iterations=5, # 对应文档中的重试限制
    agent_type="openai-tools", # Qwen 兼容 tool-calling 模式
)
5. 高级功能：人工介入 (Human-in-the-loop)
根据参考文档，为了确保生产环境安全，在 Agent 执行 SQL 之前可以加入人工审批环节。

5.1 审批流程设计
利用 LangGraph 的 interrupt 功能实现：

状态挂起：Agent 生成 SQL 后停止运行，将 SQL 语句推送给管理员。

人工审批：管理员输入 Approve 或修改意见。

继续执行：Agent 接收指令后运行 SQL 或根据意见修改。

提示：这在处理涉及敏感财务数据或大规模扫描的查询时非常关键。

6. 系统 Prompt 策略 (System Message)
为保证 Qwen 能够准确执行，需设定明确的 System Prompt：

你是一名专业的 MySQL 数据分析师。

严格按照提供的 Schema 进行查询，严禁凭空想象字段。

必须以 SELECT 开头，严禁执行任何修改操作。

如果执行报错，请仔细阅读错误信息并重新生成 SQL。