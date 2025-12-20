# 🤖 SQL Agent - 智能数据库查询助手

基于大语言模型的智能SQL查询系统，通过自然语言交互实现数据库查询和分析。

## ✨ 功能特点

- 🗣️ **自然语言查询**：使用中文描述需求，自动生成SQL语句
- 🧠 **智能理解**：基于通义千问大模型，理解复杂查询意图
- 📊 **结果可视化**：自动展示查询结果的表格和统计信息
- 🔄 **多轮对话**：支持连续对话，上下文理解
- 🛡️ **安全防护**：限制查询结果数量，防止误操作
- ☁️ **云端部署**：支持Streamlit Cloud一键部署

## 🚀 快速开始

### 在线体验
访问：[你的Streamlit应用地址]

### 本地运行

1. **克隆项目**
```bash
git clone https://github.com/你的用户名/你的仓库名.git
cd 你的仓库名
```

2. **安装依赖**
```bash
pip install -r sqlagent/requirements.txt
```

3. **配置环境变量**

创建 `.streamlit/secrets.toml` 文件：
```toml
DASHSCOPE_API_KEY = "你的API密钥"
DB_USER = "数据库用户名"
DB_PASSWORD = "数据库密码"
DB_HOST = "数据库主机"
DB_PORT = "数据库端口"
DB_NAME = "数据库名称"
```

4. **启动应用**
```bash
streamlit run sqlagent/web_ui.py
```

访问 http://localhost:8501

## 💡 使用示例

### 查询示例
- "显示前10个客户的信息"
- "统计每个产品的交易总额"
- "查找风险等级为高的所有投资组合"
- "计算上个月的总交易量"

### 功能演示
![演示截图](screenshot.png)

## 📦 项目结构

```
.
├── sqlagent/                 # 核心代码
│   ├── web_ui.py            # Streamlit Web界面
│   ├── agent.py             # SQL Agent核心
│   ├── config.py            # 配置管理
│   ├── security.py          # 安全模块
│   └── requirements.txt     # 依赖列表
├── database_create/         # 数据库初始化
│   └── migrate_to_cloud.py # 云端迁移脚本
├── .gitignore              # Git忽略配置
└── README.md               # 项目说明
```

## 🗄️ 数据库说明

项目使用云端MySQL数据库，包含以下数据表：

| 表名 | 说明 | 记录数 |
|-----|------|-------|
| managers | 管理者信息 | 500 |
| clients | 客户信息 | 15,000 |
| products | 产品信息 | 200 |
| counterparties | 交易对手 | 100 |
| portfolios | 投资组合 | 20,000 |
| transactions | 交易记录 | 100,000 |
| holdings | 持仓信息 | 106,198 |
| risk_metrics | 风险指标 | 100,000 |

**总计：341,998条数据记录**

## 🛠️ 技术栈

- **前端框架**：Streamlit
- **大语言模型**：通义千问 (Qwen)
- **数据库**：MySQL 8.4
- **AI框架**：LangChain + LangGraph
- **ORM**：SQLAlchemy
- **部署平台**：Streamlit Cloud

## 🔐 安全特性

- SQL注入防护
- 查询结果数量限制
- 敏感操作提示
- 配置信息加密存储

## 📝 开发说明

### 环境要求
- Python 3.10+
- MySQL 8.0+

### 开发模式
```bash
# 启用详细日志
export VERBOSE=True

# 启用人工确认模式
export ENABLE_HUMAN_IN_LOOP=True

streamlit run sqlagent/web_ui.py
```

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

## 👨‍💻 作者

[你的名字]

## 🙏 致谢

- [LangChain](https://github.com/langchain-ai/langchain)
- [Streamlit](https://streamlit.io/)
- [通义千问](https://tongyi.aliyun.com/)
- [SQLPub](https://sqlpub.com/)

---

⭐ 如果这个项目对你有帮助，请给个Star！

