"""
SQL 安全过滤模块
防止 SQL 注入和意外删除操作
"""
import re
from typing import Optional


MAX_HARD_LIMIT = 20


def sanitize_sql_query(query: str, default_limit: int = 20, hard_limit: int = MAX_HARD_LIMIT) -> str:
    """
    清理和验证 SQL 查询语句
    
    Args:
        query: 原始 SQL 查询
        default_limit: 默认限制行数
    
    Returns:
        清理后的 SQL 查询
    
    Raises:
        ValueError: 如果检测到非查询语句
    """
    if not query or not query.strip():
        raise ValueError("SQL 查询不能为空")
    
    # 去除首尾空白
    query = query.strip()
    
    # 禁止 DML/DDL 关键字
    deny_pattern = r"\b(INSERT|UPDATE|DELETE|ALTER|DROP|CREATE|REPLACE|TRUNCATE|GRANT|REVOKE)\b"
    if re.search(deny_pattern, query, re.IGNORECASE):
        raise ValueError("检测到非查询语句，仅允许 SELECT 操作。")
    
    # 确保是 SELECT 语句
    if not query.upper().startswith("SELECT"):
        raise ValueError("仅允许 SELECT 查询语句")
    
    # 移除末尾的分号（如果有）
    query = query.rstrip(';').strip()
    
    # 计算本次有效限制：用户给的 default_limit 也不允许超过 hard_limit
    effective_limit = min(int(default_limit), int(hard_limit))

    # 检查是否已有 LIMIT
    if "LIMIT" not in query.upper():
        # 检查是否有 ORDER BY，如果有则在 ORDER BY 之后添加 LIMIT
        order_by_match = re.search(r"\bORDER\s+BY\b", query, re.IGNORECASE)
        if order_by_match:
            # 在 ORDER BY 子句后添加 LIMIT
            query = query + f" LIMIT {effective_limit}"
        else:
            # 如果没有 ORDER BY，直接在末尾添加 LIMIT
            query = query + f" LIMIT {effective_limit}"
    else:
        # 如果已有 LIMIT，强制不超过 hard_limit（且不超过 effective_limit）
        # 支持 LIMIT n / LIMIT offset, n / LIMIT n OFFSET offset
        m1 = re.search(r"(?is)\bLIMIT\s+(\d+)\s*,\s*(\d+)\b", query)
        if m1:
            offset = int(m1.group(1))
            count = int(m1.group(2))
            count2 = min(count, effective_limit)
            query = re.sub(r"(?is)\bLIMIT\s+\d+\s*,\s*\d+\b", f"LIMIT {offset}, {count2}", query)
        else:
            m2 = re.search(r"(?is)\bLIMIT\s+(\d+)\s+OFFSET\s+(\d+)\b", query)
            if m2:
                count = int(m2.group(1))
                offset = int(m2.group(2))
                count2 = min(count, effective_limit)
                query = re.sub(r"(?is)\bLIMIT\s+\d+\s+OFFSET\s+\d+\b", f"LIMIT {count2} OFFSET {offset}", query)
            else:
                m3 = re.search(r"(?is)\bLIMIT\s+(\d+)\b", query)
                if m3:
                    count = int(m3.group(1))
                    count2 = min(count, effective_limit)
                    query = re.sub(r"(?is)\bLIMIT\s+\d+\b", f"LIMIT {count2}", query, count=1)
    
    return query + ";"


def validate_sql_syntax(query: str) -> bool:
    """
    基本 SQL 语法验证
    
    Args:
        query: SQL 查询语句
    
    Returns:
        是否通过验证
    """
    # 基本括号匹配检查
    if query.count('(') != query.count(')'):
        return False
    
    # 基本引号匹配检查
    single_quotes = query.count("'")
    double_quotes = query.count('"')
    if single_quotes % 2 != 0 or double_quotes % 2 != 0:
        return False
    
    return True

