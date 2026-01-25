"""
配置文件模块
支持从环境变量和Streamlit secrets读取配置
"""
import os
from typing import Optional

# 尝试导入 Streamlit（如果在 Streamlit 环境中运行）
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

def get_config_value(key: str, default: any = None) -> any:
    """
    从环境变量或 Streamlit secrets 获取配置值
    优先级：环境变量 > Streamlit secrets > 默认值
    """
    # 先尝试从环境变量获取
    env_value = os.getenv(key)
    if env_value is not None:
        return env_value
    
    # 如果在 Streamlit 环境中，尝试从 secrets 获取
    if HAS_STREAMLIT:
        try:
            if key in st.secrets:
                return st.secrets[key]
        except:
            pass
    
    # 返回默认值
    return default

class Config:
    """SQL Agent 配置类"""
    
    # Qwen API 配置
    DASHSCOPE_API_KEY: str = get_config_value("DASHSCOPE_API_KEY", "sk-fdefbafb8ecf480b8b0faeb3de8746fc")
    
    # ========== 数据库配置 ==========
    
    # --- 本地 Docker MySQL (备用，取消注释切换) ---
    # DB_USER: str = get_config_value("DB_USER", "root")
    # DB_PASSWORD: str = get_config_value("DB_PASSWORD", "123456")
    # DB_HOST: str = get_config_value("DB_HOST", "127.0.0.1")
    # DB_PORT: int = int(get_config_value("DB_PORT", "3306"))
    # DB_NAME: Optional[str] = get_config_value("DB_NAME", None)  # 本地数据库，启动时选择
    
    # --- 云端 MySQL (当前使用) ---
    DB_USER: str = get_config_value("DB_USER", "bobo11")
    DB_PASSWORD: str = get_config_value("DB_PASSWORD", "ls0OmCgVJIXHwawv")
    DB_HOST: str = get_config_value("DB_HOST", "mysql2.sqlpub.com")
    DB_PORT: int = int(get_config_value("DB_PORT", "3307"))
    DB_NAME: Optional[str] = get_config_value("DB_NAME", "wutongbei")  # 云端数据库名
    
    # LLM 配置
    MODEL_NAME: str = get_config_value("MODEL_NAME", "qwen3-max")
    TEMPERATURE: float = float(get_config_value("TEMPERATURE", "0"))
    STREAMING: bool = str(get_config_value("STREAMING", "False")).lower() == "true"
    
    # Agent 配置
    MAX_ITERATIONS: int = int(get_config_value("MAX_ITERATIONS", "20"))
    VERBOSE: bool = str(get_config_value("VERBOSE", "True")).lower() == "true"
    AGENT_TYPE: str = get_config_value("AGENT_TYPE", "openai-tools")
    
    # 安全配置
    DEFAULT_LIMIT: int = int(get_config_value("DEFAULT_LIMIT", "20"))
    ENABLE_HUMAN_IN_LOOP: bool = str(get_config_value("ENABLE_HUMAN_IN_LOOP", "False")).lower() == "true"
    
    @classmethod
    def get_db_uri(cls, db_name: Optional[str] = None) -> str:
        """
        获取数据库连接URI（带连接池配置）
        
        连接池参数说明：
        - pool_pre_ping=true: 连接使用前先测试，自动处理断线
        - pool_size=5: 连接池大小
        - pool_recycle=3600: 1小时回收连接，防止超时
        - max_overflow=10: 最大溢出连接数
        """
        target_db = db_name or cls.DB_NAME
        if target_db is None:
            raise ValueError("数据库名称未指定，请设置 DB_NAME 环境变量或传入 db_name 参数")
        
        # 基础连接字符串
        base_uri = f"mysql+pymysql://{cls.DB_USER}:{cls.DB_PASSWORD}@{cls.DB_HOST}:{cls.DB_PORT}/{target_db}"
        
        # 添加连接池参数（通过 engine_options 在 SQLDatabase.from_uri 中使用）
        return base_uri + "?charset=utf8mb4"
    
    @classmethod
    def get_available_databases(cls) -> list:
        """获取可用的数据库列表"""
        from sqlalchemy import create_engine, text
        engine = create_engine(
            f"mysql+pymysql://{cls.DB_USER}:{cls.DB_PASSWORD}@{cls.DB_HOST}:{cls.DB_PORT}/"
        )
        with engine.connect() as conn:
            result = conn.execute(text("SHOW DATABASES"))
            databases = [row[0] for row in result if row[0] not in ['information_schema', 'mysql', 'performance_schema', 'sys']]
        return databases

