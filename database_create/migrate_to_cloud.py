"""
将financial_asset_management数据迁移到云端MySQL数据库
作者：AI助手
日期：2025-12-20
"""
import pandas as pd
from sqlalchemy import create_engine, text, inspect
import warnings
import os
warnings.filterwarnings('ignore')

# 云端数据库连接信息
CLOUD_DB_CONFIG = {
    'host': 'mysql2.sqlpub.com',
    'port': 3307,
    'user': 'bobo11',
    'password': 'ls0OmCgVJIXHwawv',
    'database': 'wutongbei',
    'charset': 'utf8mb4'
}

def create_tables_from_structure(excel_structure_path, cloud_engine):
    """根据数据表结构.xlsx创建表结构"""
    print(f"\n{'='*60}")
    print("第一步：创建表结构")
    print(f"{'='*60}\n")
    
    # 读取表结构文件
    try:
        df = pd.read_excel(excel_structure_path)
        print(f"✅ 成功读取表结构文件，共 {len(df)} 行定义")
    except Exception as e:
        print(f"❌ 读取表结构文件失败: {e}")
        return False
    
    # 只处理financial_asset_management数据库的表
    df_financial = df[df['库名英文'] == 'financial_asset_management']
    
    if df_financial.empty:
        print("❌ 未找到financial_asset_management数据库的表定义")
        return False
    
    unique_tables = df_financial['表英文'].unique()
    print(f"📊 需要创建 {len(unique_tables)} 个表: {', '.join(unique_tables)}\n")
    
    with cloud_engine.connect() as conn:
        for table_name in unique_tables:
            # 检查表是否已存在
            inspector = inspect(cloud_engine)
            if table_name in inspector.get_table_names():
                print(f"  ⚠ 表 {table_name} 已存在，先删除...")
                try:
                    conn.execute(text(f"DROP TABLE IF EXISTS `{table_name}`"))
                    conn.commit()
                except Exception as e:
                    print(f"    删除表失败: {e}")
            
            # 获取表信息
            table_info = df_financial[df_financial['表英文'] == table_name].iloc[0]
            table_desc = table_info.get('表描述', '') if '表描述' in table_info else ''
            
            # 获取该表的所有字段
            table_fields = df_financial[df_financial['表英文'] == table_name]
            
            # 先找出主键字段
            primary_key_field = None
            for _, field_row in table_fields.iterrows():
                field_name = field_row['字段英文名']
                field_comment = str(field_row.get('中文注释', ''))
                if '主键' in field_comment:
                    primary_key_field = field_name
                    break
            
            # 构建CREATE TABLE语句
            create_table_sql = f"CREATE TABLE `{table_name}` (\n"
            
            columns_sql = []
            for _, field_row in table_fields.iterrows():
                field_name = field_row['字段英文名']
                field_chinese = field_row.get('字段中文名', '')
                field_comment = field_row.get('中文注释', '')
                
                # 数据类型映射
                field_lower = field_name.lower()
                if field_name == primary_key_field:
                    col_def = f"  `{field_name}` INT AUTO_INCREMENT PRIMARY KEY"
                elif field_lower.startswith('is_') or field_lower.startswith('has_'):
                    col_def = f"  `{field_name}` TINYINT(1) DEFAULT 0"
                elif 'time' in field_lower or 'date' in field_lower:
                    col_def = f"  `{field_name}` DATETIME"
                elif 'amount' in field_lower or 'price' in field_lower or 'value' in field_lower or 'balance' in field_lower:
                    col_def = f"  `{field_name}` DECIMAL(15,2)"
                elif 'json' in str(field_comment).lower():
                    col_def = f"  `{field_name}` JSON"
                else:
                    col_def = f"  `{field_name}` VARCHAR(255)"
                
                # 添加注释
                if field_chinese or field_comment:
                    comment = f"{field_chinese}"
                    if field_comment and pd.notna(field_comment):
                        comment += f": {field_comment}"
                    # 转义单引号
                    comment = comment.replace("'", "\\'")
                    col_def += f" COMMENT '{comment}'"
                
                columns_sql.append(col_def)
            
            create_table_sql += ',\n'.join(columns_sql)
            create_table_sql += f"\n) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci"
            
            if table_desc and pd.notna(table_desc):
                table_desc = str(table_desc).replace("'", "\\'")
                create_table_sql += f" COMMENT='{table_desc}'"
            
            try:
                conn.execute(text(create_table_sql))
                conn.commit()
                print(f"  ✅ 创建表: {table_name}")
            except Exception as e:
                print(f"  ❌ 创建表 {table_name} 失败: {e}")
                return False
    
    print(f"\n✅ 表结构创建完成！\n")
    return True

def import_data_to_cloud(excel_data_path, cloud_engine):
    """导入Excel数据到云端数据库"""
    print(f"\n{'='*60}")
    print("第二步：导入数据")
    print(f"{'='*60}\n")
    
    # 读取Excel文件的所有sheet
    try:
        excel_file = pd.ExcelFile(excel_data_path)
        print(f"📂 Excel文件: {os.path.basename(excel_data_path)}")
        print(f"📊 Sheet数量: {len(excel_file.sheet_names)}")
        print(f"📋 Sheet列表: {', '.join(excel_file.sheet_names)}\n")
    except Exception as e:
        print(f"❌ 读取Excel文件失败: {e}")
        return False
    
    # 遍历每个sheet
    success_count = 0
    for sheet_name in excel_file.sheet_names:
        try:
            # 读取sheet数据
            df = pd.read_excel(excel_data_path, sheet_name=sheet_name)
            
            if df.empty:
                print(f"  ⚠ 表 '{sheet_name}' 为空，跳过")
                continue
            
            print(f"  📊 处理表: {sheet_name} ({len(df)} 行数据)")
            
            # 检查表是否存在
            inspector = inspect(cloud_engine)
            if sheet_name not in inspector.get_table_names():
                print(f"    ⚠ 表 '{sheet_name}' 不存在，跳过")
                continue
            
            # 清理数据：处理NaN值
            df = df.replace({pd.NA: None, 'nan': None, 'NaN': None})
            df = df.where(pd.notnull(df), None)
            
            # 批量插入数据
            batch_size = 500  # 云端数据库使用较小的批次
            total_rows = len(df)
            inserted_rows = 0
            
            for i in range(0, total_rows, batch_size):
                batch_df = df.iloc[i:i+batch_size]
                
                try:
                    # 使用pandas的to_sql方法批量插入
                    batch_df.to_sql(
                        name=sheet_name,
                        con=cloud_engine,
                        if_exists='append',
                        index=False,
                        method='multi',
                        chunksize=100
                    )
                    inserted_rows += len(batch_df)
                    if i + batch_size >= total_rows:
                        print(f"    ✅ 进度: {inserted_rows}/{total_rows} 行")
                except Exception as e:
                    print(f"    ⚠ 批量插入失败 (行 {i}-{i+len(batch_df)}): {e}")
                    # 尝试逐行插入
                    for idx, row in batch_df.iterrows():
                        try:
                            row_df = pd.DataFrame([row.to_dict()])
                            row_df.to_sql(
                                name=sheet_name,
                                con=cloud_engine,
                                if_exists='append',
                                index=False
                            )
                            inserted_rows += 1
                        except Exception as row_error:
                            print(f"      跳过问题行 {idx}: {row_error}")
            
            print(f"    ✅ 成功导入 {inserted_rows}/{total_rows} 行数据\n")
            success_count += 1
                
        except Exception as e:
            print(f"  ❌ 处理Sheet '{sheet_name}' 时出错: {e}\n")
            continue
    
    print(f"✅ 数据导入完成！成功导入 {success_count}/{len(excel_file.sheet_names)} 个表\n")
    return success_count > 0

def main():
    """主函数"""
    print(f"\n{'#'*60}")
    print("# 财务资产管理数据云端迁移工具")
    print("# Financial Asset Management Data Migration to Cloud")
    print(f"{'#'*60}\n")
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 文件路径
    structure_file = os.path.join(script_dir, '数据表结构.xlsx')
    data_file = os.path.join(script_dir, 'financial_asset_management.xlsx')
    
    # 检查文件是否存在
    if not os.path.exists(structure_file):
        print(f"❌ 表结构文件不存在: {structure_file}")
        return
    
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return
    
    # 创建云端数据库连接
    connection_string = (
        f"mysql+pymysql://{CLOUD_DB_CONFIG['user']}:{CLOUD_DB_CONFIG['password']}"
        f"@{CLOUD_DB_CONFIG['host']}:{CLOUD_DB_CONFIG['port']}/{CLOUD_DB_CONFIG['database']}"
        f"?charset={CLOUD_DB_CONFIG['charset']}"
    )
    
    print(f"🌐 连接云端数据库...")
    print(f"   主机: {CLOUD_DB_CONFIG['host']}:{CLOUD_DB_CONFIG['port']}")
    print(f"   数据库: {CLOUD_DB_CONFIG['database']}")
    print(f"   用户: {CLOUD_DB_CONFIG['user']}")
    
    try:
        cloud_engine = create_engine(connection_string, pool_pre_ping=True)
        # 测试连接
        with cloud_engine.connect() as conn:
            result = conn.execute(text("SELECT DATABASE()"))
            db_name = result.fetchone()[0]
            print(f"✅ 连接成功！当前数据库: {db_name}\n")
    except Exception as e:
        print(f"❌ 连接云端数据库失败: {e}")
        return
    
    # 第一步：创建表结构
    if not create_tables_from_structure(structure_file, cloud_engine):
        print("❌ 表结构创建失败，停止导入")
        return
    
    # 第二步：导入数据
    if not import_data_to_cloud(data_file, cloud_engine):
        print("❌ 数据导入失败")
        return
    
    print(f"\n{'#'*60}")
    print("# ✅ 迁移完成！")
    print("# 你的数据已成功导入到云端数据库")
    print(f"{'#'*60}\n")

if __name__ == '__main__':
    main()

