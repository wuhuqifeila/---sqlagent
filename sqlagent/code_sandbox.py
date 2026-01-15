"""
代码沙箱模块
提供安全的代码执行环境，用于执行大模型生成的数据分析代码
"""
import os
import asyncio
import time
from typing import Dict, Any
from jupyter_client import KernelManager
import pandas as pd


class CodeSandbox:
    """基于 Jupyter Kernel 的代码沙箱执行器"""
    
    def __init__(self, timeout: int = 60, max_output_size: int = 1024 * 1024):
        """
        初始化代码沙箱
        
        Args:
            timeout: 代码执行超时时间（秒）
            max_output_size: 最大输出大小（字节）
        """
        self.timeout = timeout
        self.max_output_size = max_output_size
        self.km = None
        self.kc = None
        self.kernel_started = False
    
    def start_kernel(self) -> bool:
        """
        启动 Jupyter 内核
        
        Returns:
            是否成功启动
        """
        try:
            self.km = KernelManager()
            self.km.start_kernel()
            self.kc = self.km.client()
            self.kc.start_channels()
            self.kc.wait_for_ready(timeout=self.timeout)
            self.kernel_started = True
            return True
        except Exception as e:
            print(f"启动内核失败: {e}")
            return False
    
    def stop_kernel(self):
        """停止 Jupyter 内核"""
        if self.kc:
            try:
                self.kc.stop_channels()
            except Exception:
                pass
            self.kc = None
        
        if self.km:
            try:
                self.km.shutdown_kernel(now=True)
            except Exception:
                pass
            self.km = None
        
        self.kernel_started = False
    
    async def execute_code_with_timeout(self, code: str, df_path: str) -> Dict[str, Any]:
        """
        在沙箱中执行代码
        
        Args:
            code: 要执行的 Python 代码
            df_path: Excel 文件路径（包含查询结果数据）
        
        Returns:
            执行结果字典
        """
        if not self.kernel_started:
            return {"success": False, "error": "内核未启动"}
        
        try:
            # 设置超时
            start_time = time.time()
            
            # 1. 注入数据到内核
            setup_code = f"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 配置matplotlib英文字体（确保英文显示）
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
matplotlib.rcParams['axes.unicode_minus'] = False

# 中文到英文的简单映射（用于图表标签）
CHINESE_TO_ENGLISH_MAPPING = {{
    '类别': 'Category',
    '分类': 'Classification', 
    '数值': 'Value',
    '数量': 'Count',
    '金额': 'Amount',
    '客户': 'Customer',
    '用户': 'User',
    '产品': 'Product',
    '日期': 'Date',
    '时间': 'Time',
    '名称': 'Name',
    '类型': 'Type',
    '状态': 'Status',
    '风险': 'Risk',
    '等级': 'Level',
    '资产': 'Asset',
    '总计': 'Total',
    '平均': 'Average',
    '最大': 'Max',
    '最小': 'Min',
    '测试': 'Test',
    '标签': 'Label',
    '系列': 'Series',
    '组': 'Group',
    '项目': 'Item',
    '指标': 'Metric',
    '分析': 'Analysis',
    '报告': 'Report',
    '数据': 'Data',
    '结果': 'Result',
    '行': 'Row',
    '列': 'Column'
}}

def translate_chinese_to_english(text):
    \"\"\"
    将简单的中文文本转换为英文
    对于复杂文本，返回原始文本（可能包含数字或其他字符）
    \"\"\"
    if not isinstance(text, str):
        return str(text)
    
    # 如果文本主要是数字或英文，直接返回
    if all(c.isascii() or c.isdigit() or c in ' .,-_' for c in text):
        return text
    
    # 简单的中文到英文替换
    translated = text
    for chinese, english in CHINESE_TO_ENGLISH_MAPPING.items():
        translated = translated.replace(chinese, english)
    
    # 如果翻译后没有变化，且包含中文字符，生成通用英文标签
    if translated == text and any('\\u4e00' <= c <= '\\u9fff' for c in text):
        # 提取数字部分或生成通用标签
        import re
        numbers = re.findall(r'\\d+', text)
        if numbers:
            return f'Item_{{numbers[0]}}'
        else:
            # 为纯中文文本生成通用英文标签
            return f'Label_{{hash(text) % 1000}}'
    
    return translated

def safe_translate_labels(labels):
    \"\"\"
    安全地转换标签列表
    \"\"\"
    if labels is None:
        return labels
    try:
        return [translate_chinese_to_english(label) for label in labels]
    except Exception:
        return labels

# 加载数据
df = pd.read_excel(r'{df_path}')
print(f"数据加载成功，形状: {{df.shape}}")

# 重写matplotlib的绘图函数以自动转换标签（可选）
# 这里提供辅助函数供分析代码使用
print("已加载中文到英文转换功能，可在绘图时使用 safe_translate_labels() 函数")
"""
            
            # 执行数据注入
            msg_id = self.kc.execute(setup_code)
            setup_result = await self._get_execution_result(msg_id, start_time)
            
            if not setup_result["success"]:
                return {"success": False, "error": f"数据注入失败: {setup_result.get('error', '未知错误')}"}
            
            # 2. 执行用户代码
            msg_id = self.kc.execute(code)
            result = await self._get_execution_result(msg_id, start_time)
            
            return result
            
        except Exception as e:
            return {"success": False, "error": f"执行异常: {str(e)}"}
    
    async def _get_execution_result(self, msg_id: str, start_time: float) -> Dict[str, Any]:
        """获取代码执行结果"""
        stdout = ""
        stderr = ""
        error_occurred = False
        
        try:
            while True:
                # 检查超时
                if time.time() - start_time > self.timeout:
                    return {"success": False, "error": "代码执行超时"}
                
                # 获取消息
                msg = None
                if self.kc.shell_channel.msg_ready():
                    msg = self.kc.shell_channel.get_msg()
                elif self.kc.iopub_channel.msg_ready():
                    msg = self.kc.iopub_channel.get_msg()
                elif self.kc.stdin_channel.msg_ready():
                    msg = self.kc.stdin_channel.get_msg()
                
                if msg is None:
                    # 短暂等待避免CPU占用过高
                    await asyncio.sleep(0.1)
                    continue
                
                msg_type = msg['msg_type']
                content = msg['content']
                
                if msg_type == 'stream':
                    if content.get('name') == 'stdout':
                        stdout += content.get('text', '')
                    elif content.get('name') == 'stderr':
                        stderr += content.get('text', '')
                elif msg_type == 'error':
                    error_occurred = True
                    ename = content.get('ename', '')
                    evalue = content.get('evalue', '')
                    traceback = '\n'.join(content.get('traceback', []))
                    stderr += f"{ename}: {evalue}\n{traceback}"
                elif msg_type == 'execute_reply':
                    if msg['parent_header'].get('msg_id') == msg_id:
                        break
                
                # 检查输出大小限制
                if len(stdout) + len(stderr) > self.max_output_size:
                    return {"success": False, "error": "输出过大，已截断"}
            
            if error_occurred:
                return {"success": False, "error": stderr, "stdout": stdout}
            else:
                return {"success": True, "stdout": stdout, "stderr": stderr}
                
        except Exception as e:
            return {"success": False, "error": f"获取结果失败: {str(e)}"}


async def execute_analysis_code(
    analysis_code: str, 
    df_path: str, 
    pdf_output_path: str,
    timeout: int = 60
) -> Dict[str, Any]:
    """
    执行数据分析代码并生成PDF报告
    
    Args:
        analysis_code: 大模型生成的数据分析代码
        df_path: 输入Excel文件路径
        pdf_output_path: 输出PDF文件路径
        timeout: 执行超时时间
    
    Returns:
        执行结果字典
    """
    sandbox = CodeSandbox(timeout=timeout)
    
    try:
        # 启动内核
        if not sandbox.start_kernel():
            return {"success": False, "error": "无法启动代码执行环境"}
        
        # 构建完整的执行代码（确保生成PDF）
        complete_code = f"""
# 确保生成PDF报告
import os
import sys
import traceback

# 执行用户提供的分析代码
analysis_success = False
analysis_error = None
try:
{analysis_code.replace(chr(10), chr(10) + '    ')}
    analysis_success = True
except Exception as e:
    analysis_error = str(e)
    print(f"分析代码执行出错: {{e}}")
    traceback.print_exc()

# 确保PDF文件被创建
pdf_path = r'{pdf_output_path}'
pdf_created = False

# 检查用户代码是否已经创建了PDF
if os.path.exists(pdf_path):
    pdf_created = True
    print(f"用户代码已成功生成PDF报告: {{pdf_path}}")
else:
    # 用户代码没有生成PDF，我们创建一个包含详细文字描述的完整报告
    try:
        import pandas as pd
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.pdfbase import pdfmetrics
        
        # 加载数据用于报告生成
        df = pd.read_excel(r'{df_path}')
        
        # 设置英文字体（确保兼容性）
        font_name = 'Helvetica'
        try:
            from reportlab.pdfbase.ttfonts import TTFont
            
            # 尝试注册系统字体
            font_paths_to_try = [
                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
                '/usr/share/fonts/truetype/droid/DroidSans.ttf'
            ]
            
            for font_path in font_paths_to_try:
                try:
                    if os.path.exists(font_path):
                        pdfmetrics.registerFont(TTFont('ReportFont', font_path))
                        font_name = 'ReportFont'
                        print(f"成功注册字体: {{font_path}}")
                        break
                except Exception as font_e:
                    continue
                    
        except Exception as e:
            print(f"字体注册失败，使用默认字体: {{e}}")
        
        # 创建PDF文档
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        styles = getSampleStyleSheet()
        
        # 创建自定义样式
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            fontName=font_name,
            spaceAfter=30,
            alignment=1
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            fontName=font_name,
            spaceAfter=12,
            spaceBefore=20
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontName=font_name,
            fontSize=10,
            leading=14,
            spaceAfter=8
        )
        
        # 构建报告内容
        story = []
        story.append(Paragraph("数据分析报告", title_style))
        story.append(Spacer(1, 12))
        
        # 数据概览
        story.append(Paragraph("1. 数据概览", heading_style))
        story.append(Paragraph(f"数据集包含 {{df.shape[0]}} 行和 {{df.shape[1]}} 列。", normal_style))
        story.append(Paragraph(f"列名: {{', '.join(df.columns.tolist())}}", normal_style))
        
        # 数据类型信息
        story.append(Paragraph("2. 数据类型", heading_style))
        dtypes_info = []
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            dtypes_info.append(f"• {{col}}: {{dtype_str}}")
        story.append(Paragraph("\\n".join(dtypes_info), normal_style))
        
        # 缺失值分析
        story.append(Paragraph("3. 缺失值分析", heading_style))
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        missing_info = []
        total_missing = missing_counts.sum()
        if total_missing > 0:
            story.append(Paragraph(f"缺失值总数: {{total_missing}} (占总数据的 {{total_missing / (df.shape[0] * df.shape[1]) * 100:.2f}}%)", normal_style))
            for col in df.columns:
                if missing_counts[col] > 0:
                    missing_info.append(f"• {{col}}: {{missing_counts[col]}} 个缺失值 ({{missing_percentages[col]:.2f}}%)")
            if missing_info:
                story.append(Paragraph("\\n".join(missing_info), normal_style))
        else:
            story.append(Paragraph("数据集中未发现缺失值。", normal_style))
        
        # 基本统计摘要（仅数值列）
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            story.append(Paragraph("4. 基本统计摘要", heading_style))
            story.append(Paragraph("下表显示了数值列的描述性统计信息：", normal_style))
            
            # 创建统计表格
            stats_df = df[numeric_cols].describe()
            # 转换为表格数据
            table_data = [['统计量'] + [str(col) for col in numeric_cols]]
            for stat in stats_df.index:
                row = [str(stat)] + [f"{{val:.4f}}" if isinstance(val, float) else str(val) for val in stats_df.loc[stat]]
                table_data.append(row)
            
            # 创建表格
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), font_name),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
        
        # 分类列分析
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            story.append(Paragraph("5. 分类列分析", heading_style))
            story.append(Paragraph("分类列的唯一值数量：", normal_style))
            cat_info = []
            for col in categorical_cols[:5]:  # 限制显示前5个分类列
                unique_count = df[col].nunique()
                cat_info.append(f"• {{col}}: {{unique_count}} 个唯一值")
            if len(categorical_cols) > 5:
                cat_info.append(f"... 还有 {{len(categorical_cols) - 5}} 个分类列")
            story.append(Paragraph("\\n".join(cat_info), normal_style))
        
        # 执行状态信息
        story.append(Paragraph("6. 执行状态", heading_style))
        if analysis_success:
            story.append(Paragraph("分析代码执行成功，但用户代码未生成PDF文件。", normal_style))
        else:
            story.append(Paragraph("分析代码执行失败。此备用报告包含基本数据信息。", normal_style))
            if analysis_error:
                story.append(Paragraph(f"错误信息: {{analysis_error}}", normal_style))
        
        story.append(Spacer(1, 20))
        story.append(Paragraph("注意: 这是一个备用报告，因为原始分析代码未能生成PDF文件。", normal_style))
        
        # 构建PDF
        doc.build(story)
        pdf_created = True
        print(f"已创建详细的fallback PDF报告: {{pdf_path}}")
        
    except Exception as e:
        print(f"创建fallback PDF报告失败: {{e}}")
        traceback.print_exc()

if pdf_created:
    print(f"PDF报告生成成功: {{pdf_path}}")
else:
    print("PDF报告生成失败")
"""
        
        # 执行代码
        result = await sandbox.execute_code_with_timeout(complete_code, df_path)
        
        # 检查PDF是否生成成功
        if result["success"] and os.path.exists(pdf_output_path):
            result["pdf_path"] = pdf_output_path
        else:
            result["success"] = False
            if "error" not in result:
                result["error"] = "PDF报告生成失败"
        
        return result
        
    finally:
        sandbox.stop_kernel()