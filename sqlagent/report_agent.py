"""
Report Agent 模块
专门负责数据分析报告的生成，与数据库查询Agent分离
"""
import os
import asyncio
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
from langchain_core.messages import HumanMessage

from .config import Config
from .code_sandbox import execute_analysis_code


class ReportAgent:
    """独立的报告生成Agent，专门负责数据分析报告的生成"""
    
    def __init__(self, model_name: Optional[str] = None, temperature: Optional[float] = None):
        """
        初始化 Report Agent
        
        Args:
            model_name: 模型名称
            temperature: 温度参数
        """
        from langchain_community.chat_models import ChatTongyi
        self.llm = ChatTongyi(
            model_name=model_name or Config.MODEL_NAME,
            temperature=temperature if temperature is not None else Config.TEMPERATURE
        )
    
    def _build_analysis_prompt(
        self, 
        question: str, 
        custom_analysis_req: Optional[str] = None,
        df_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        构建数据分析提示词
        
        Args:
            question: 原始用户问题
            custom_analysis_req: 自定义分析要求
            df_info: DataFrame信息（可选）
        
        Returns:
            分析提示词
        """
        base_prompt = f"""
你是一名专业的数据科学家。请基于用户的问题和提供的DataFrame（变量名: df）生成完整的Python数据分析代码。

用户问题: {question}
"""
        
        if custom_analysis_req and custom_analysis_req.strip():
            base_prompt += f"\n用户自定义分析要求: {custom_analysis_req.strip()}\n"
        
        if df_info:
            base_prompt += f"\n数据信息:\n- 形状: {df_info.get('shape', '未知')}\n"
            base_prompt += f"- 列名: {', '.join(df_info.get('columns', []))}\n"
            base_prompt += f"- 数值列: {', '.join(df_info.get('numeric_columns', []))}\n"
            base_prompt += f"- 分类列: {', '.join(df_info.get('categorical_columns', []))}\n"
        
        analysis_prompt = base_prompt + f"""
要求：
1. 进行全面的数据分析，包括基本统计、相关性分析、分布分析等
2. 生成高质量的可视化图表（使用matplotlib/seaborn），**所有图表标题、标签、图例必须使用英文**
3. 将所有分析结果整合到一个PDF报告中，**包含详细的中文文字描述和解释**
4. PDF报告必须保存到指定路径
5. **中文文字描述内容应包括**：
   - 数据概览（形状、列信息、数据类型）
   - 缺失值分析
   - 基本统计摘要
   - 数值列的分布特征分析
   - 分类列的频次分析  
   - 相关性分析（如果适用）
   - 关键发现和洞察总结
6. 不要使用plt.show()，直接保存图表到PDF
7. 确保代码完整可运行，包含所有必要的import语句
8. **重要：处理中文标签时使用提供的转换函数**：
   ```python
   # 如果您的数据包含中文分类标签，使用以下函数转换为英文
   # x_labels = safe_translate_labels(original_labels)
   # ax.set_xticklabels(x_labels)
   
   # 或者在创建图表时直接转换
   # categories = [translate_chinese_to_english(cat) for cat in original_categories]
   ```
9. **必须包含matplotlib英文配置**：
   ```python
   import matplotlib.pyplot as plt
   import matplotlib
   matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
   matplotlib.rcParams['axes.unicode_minus'] = False
   ```
10. **必须包含reportlab字体配置（使用最安全的实现）**：
    ```python
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    import os
    
    # 安全的字体配置 - 确保font_name_for_pdf始终有默认值
    font_name_for_pdf = 'Helvetica'  # 默认字体，确保变量始终定义
    
    try:
        # 尝试注册中文字体（仅Linux路径）
        chinese_font_registered = False
        font_paths = [
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        ]
        
        for font_path_candidate in font_paths:
            if os.path.exists(font_path_candidate):
                try:
                    pdfmetrics.registerFont(TTFont('ChineseFont', font_path_candidate))
                    font_name_for_pdf = 'ChineseFont'
                    chinese_font_registered = True
                    break
                except Exception:
                    continue
        
        # 如果没有成功注册中文字体，保持默认的Helvetica
        if not chinese_font_registered:
            font_name_for_pdf = 'Helvetica'
            
    except Exception:
        # 任何异常都使用默认字体
        font_name_for_pdf = 'Helvetica'
    
    # 现在font_name_for_pdf总是被正确定义
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        fontName=font_name_for_pdf,
        spaceAfter=30,
        alignment=1
    )
    ```
11. 在创建PDF样式时，始终使用 fontName=font_name_for_pdf 参数
12. 确保所有图表在保存前调用 plt.savefig()，保存后调用 plt.close()
13. **重要：在函数开始处就定义font_name_for_pdf变量，确保在整个代码中都可用**

**数据特点说明：**
- 数据可能包含中文列名和文本数据
- 包含日期时间列（register_date, create_time, update_time）
- 数值列可能包括total_assets
- 需要适当处理非数值列，避免在数值分析中出错

**输出格式要求：**
- PDF报告应该包含中文文字描述段落和对应的英文图表
- 每个分析部分都应该有清晰的中文标题和详细的文字解释
- 文字描述应该专业、准确、易于理解
- **图表中的所有文字（包括坐标轴标签、图例、标题）必须是英文**

**安全要求：**
- 所有变量必须在使用前初始化
- 必须包含完整的异常处理
- 避免使用Windows或macOS特定的路径
- 确保代码在Linux环境中能正常运行

请只返回Python代码，不要包含任何解释性文字。
"""
        return analysis_prompt
    
    def _extract_code_from_response(self, response_content: str) -> str:
        """
        从模型响应中提取纯Python代码
        
        Args:
            response_content: 模型响应内容
        
        Returns:
            提取的Python代码
        """
        analysis_code = response_content.strip()
        
        # 如果代码被包裹在代码块中，提取纯代码
        if analysis_code.startswith("```python"):
            analysis_code = analysis_code[9:-3] if analysis_code.endswith("```") else analysis_code[9:]
        elif analysis_code.startswith("```"):
            analysis_code = analysis_code[3:-3] if analysis_code.endswith("```") else analysis_code[3:]
        
        return analysis_code
    
    def _get_df_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        获取DataFrame的基本信息
        
        Args:
            df: 输入的DataFrame
        
        Returns:
            DataFrame信息字典
        """
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        return {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'numeric_columns': numeric_columns,
            'categorical_columns': categorical_columns,
            'row_count': len(df),
            'column_count': len(df.columns)
        }
    
    async def generate_report(
        self,
        df: pd.DataFrame,
        question: str,
        custom_analysis_req: Optional[str] = None,
        timeout: int = 120
    ) -> Dict[str, Any]:
        """
        生成数据分析报告
        
        Args:
            df: 输入的DataFrame数据
            question: 用户问题
            custom_analysis_req: 自定义分析要求（可选）
            timeout: 执行超时时间（秒）
        
        Returns:
            报告生成结果字典，包含：
            - success: 是否成功
            - pdf_bytes: PDF文件字节（如果成功）
            - filename: 文件名
            - error: 错误信息（如果失败）
        """
        try:
            # 生成临时文件路径
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            excel_filename = f"temp_data_{timestamp}.xlsx"
            excel_path = f"/tmp/{excel_filename}"
            report_filename = f"analysis_report_{timestamp}.pdf"
            report_path = f"/tmp/{report_filename}"
            
            # 保存DataFrame到Excel
            df.to_excel(excel_path, index=False)
            
            # 获取DataFrame信息用于提示词
            df_info = self._get_df_info(df)
            
            # 构建分析提示词
            analysis_prompt = self._build_analysis_prompt(question, custom_analysis_req, df_info)
            
            # 调用大模型生成分析代码
            analysis_code_response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
            analysis_code = self._extract_code_from_response(analysis_code_response.content)
            
            # 在沙箱中执行代码
            result = await execute_analysis_code(analysis_code, excel_path, report_path, timeout=timeout)
            
            if result["success"] and os.path.exists(report_path):
                # 读取PDF文件
                with open(report_path, "rb") as f:
                    pdf_bytes = f.read()
                
                # 清理临时文件
                try:
                    os.remove(excel_path)
                    os.remove(report_path)
                except Exception:
                    pass
                
                return {
                    "success": True,
                    "pdf_bytes": pdf_bytes,
                    "filename": report_filename
                }
            else:
                error_msg = result.get("error", "未知错误")
                # 清理临时文件
                try:
                    if os.path.exists(excel_path):
                        os.remove(excel_path)
                    if os.path.exists(report_path):
                        os.remove(report_path)
                except Exception:
                    pass
                
                return {
                    "success": False,
                    "error": error_msg,
                    "analysis_code": analysis_code  # 返回生成的代码用于调试
                }
                
        except Exception as e:
            # 清理临时文件
            try:
                if 'excel_path' in locals() and os.path.exists(excel_path):
                    os.remove(excel_path)
                if 'report_path' in locals() and os.path.exists(report_path):
                    os.remove(report_path)
            except Exception:
                pass
            
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_llm(self):
        """获取LLM实例（用于外部调用）"""
        return self.llm