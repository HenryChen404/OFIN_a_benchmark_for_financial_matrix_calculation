import pandas as pd
import glob
from pathlib import Path
import camelot
import json
import aisuite as ai
import os
from typing import Dict, List, Tuple, Any
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('financial_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

# Set API keys as environment variables
os.environ['OPENAI_API_KEY'] = "YOUR_API_KEY"
os.environ['ANTHROPIC_API_KEY'] = "YOUR_API_KEY"
os.environ['GOOGLE_API_KEY'] = "YOUR_API_KEY"
os.environ['MISTRAL_API_KEY'] = "YOUR_API_KEY"

# Initialize aisuite client
client = ai.Client()

# Define models from different providers with correct identifiers

# read an excel file
ori_df = pd.read_excel(r'D:\!MSAFA\！Course\3-GenAI\Project\FI_T4.xlsx')

df = ori_df.drop([0, 1]).reset_index(drop=True)
df = df[df['Accper'] == "2023-12-31"]
df = df[df['Typrep'] == "A"]
df = df[df['Source'] == 0]

print(df)

# 创建重命名映射字典
rename_dict = {
    'F040101B': 'Accounts_Receivable_Revenue_Ratio',
    'F040202B': 'Accounts_Receivable_Turnover',
    'F040302B': 'Accounts_Receivable_Days',
    'F040401B': 'Inventory_Revenue_Ratio',
    'F040502B': 'Inventory_Turnover',
    'F040602B': 'Inventory_Days',
    'F040702B': 'Operating_Cycle',
    'F040802B': 'Accounts_Payable_Turnover',
    'F040902B': 'Working_Capital_Turnover',
    'F041002B': 'Cash_Turnover',
    'F041101B': 'Current_Assets_Revenue_Ratio',
    'F041202B': 'Current_Assets_Turnover',
    'F041301B': 'Fixed_Assets_Revenue_Ratio',
    'F041402B': 'Fixed_Assets_Turnover',
    'F041502B': 'Noncurrent_Assets_Turnover',
    'F041601B': 'Capital_Intensity',
    'F041702B': 'Total_Assets_Turnover',
    'F041802B': 'Equity_Turnover'
}

# 重命名DataFrame的列
df = df.rename(columns=rename_dict)

#df1=df.iloc[0:1]

df1 = df.sample(n=6, random_state=101)
#df1 = df[df['Stkcd'] == '000552']

matrix = [*rename_dict.values()]

df1 = df1.fillna(-101)
df1 = df1.round(2)

# 删除多个指定列
columns_to_drop = ['ShortName', 'Accper', 'Typrep','Indnme1','Source']
df1 = df1.drop(columns=columns_to_drop)
#%%
def descriptive_statistics(df):
    stats = pd.DataFrame({
        "Missing Values": df.isnull().sum(),       # 空值数
        "Mean": df.mean(),                         # 平均值
        "Variance": df.var()                       # 方差
    })
    return stats

# 执行函数
df=df.drop(columns=columns_to_drop)
df = df.drop(columns=['Stkcd'])
df = df.apply(pd.to_numeric, errors='coerce')

result = descriptive_statistics(df)

# 打印结果
print(result)
#%% Find and open the file
def check_files(Stkcd, pdf_folder):
    stkcd = str(Stkcd).zfill(6)
    
    # Create search pattern
    pattern = str(Path(pdf_folder) / f"{stkcd}*.pdf")
    
    # Find matching files
    matching_files = glob.glob(pattern)
    
    if matching_files:
        print(f"Found file for {stkcd}: {matching_files[0]}")
        return matching_files[0]
    else:
        print(f"No file found for {stkcd}")
        return False

#%% Clean Json Format
def clean_json_response(content):
    try:
        # 查找```json和```之间的内容
        json_start = content.find("```json") + 7  # 跳过```json
        json_end = content.find("```", json_start)
        
        if json_start > 6 and json_end > json_start:  # 确保找到了标记
            json_content = content[json_start:json_end].strip()
            return json_content
        
        # 如果没有找到代码块标记，尝试直接寻找JSON对象
        else:
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                return content[start:end]
                
        raise ValueError("未找到有效的JSON内容")
            
    except Exception as e:
        print(f"清理JSON时出错: {str(e)}")
        print("原始内容:", content)
        return content

#%% Find Main Statements
def find_statements_by_subjects(tables):
    """
    通过关键科目找出相关的财务报表，每类取匹配度最高的前四名
    """
    # 定义关键科目
    key_subjects = {
        "资产负债表相关": [
            "资产总计", "负债总计", "所有者权益合计",
            "货币资金", "交易性金融资产", "应收票据及应收账款", 
            "预付款项", "其他应收款", "存货", "固定资产", 
            "在建工程", "无形资产", "长期股权投资", "递延所得税资产", 
            "应付账款", "预收款项", "长期借款", "应付债券", 
            "递延所得税负债"
        ],
        "利润表相关": [
            "营业收入", "营业成本", "税金及附加", 
            "销售费用", "管理费用", "研发费用", 
            "财务费用", "其他收益", "投资收益", 
            "公允价值变动收益", "资产减值损失", 
            "信用减值损失", "营业利润", "营业外收入", 
            "营业外支出", "利润总额", "所得税费用", "净利润"
        ],
        "现金流量表相关": [
            "经营活动", "投资活动", 
            "筹资活动", "现金及现金等价物", 
            "分配股利、利润或偿付利息支付的现金"
        ]
    }

    
    # 存储每种类型的表格和匹配数
    categorized_tables = {
        "资产负债表相关": [],
        "利润表相关": [],
        "现金流量表相关": []
    }
    
    # 遍历所有表格
    for i, table in enumerate(tables):
        df = table.df
        df_str = df.to_string().lower()
        
        # 对每种类型检查匹配度
        for statement_type, subjects in key_subjects.items():
            matched_subjects = [
                subject for subject in subjects 
                if subject.lower() in df_str
            ]
            
            # 如果匹配至少3个关键科目，添加到对应类别
            if len(matched_subjects) >= 3 and df.shape[0] > 5:
                categorized_tables[statement_type].append({
                    'table_index': i,
                    'df': df,
                    'matched_count': len(matched_subjects),
                    'matched_subjects': matched_subjects,
                    'shape': df.shape
                })
    
    # 选取每类匹配度最高的前四名
    selected_tables = {}
    for category, tables_list in categorized_tables.items():
        if not tables_list:
            continue
            
        # 按匹配数量排序
        tables_list.sort(key=lambda x: x['matched_count'], reverse=True)
        
        # 获取前2个匹配数
        top_counts = []
        for t in tables_list[:1]:
            if t['matched_count'] not in top_counts:
                top_counts.append(t['matched_count'])
        
        # 选择匹配数在前4名内的表格
        min_count = min(top_counts) if top_counts else 0
        selected = [t for t in tables_list if t['matched_count'] >= min_count]
        
        # 添加到结果中
        for table_info in selected:
            name = f"{category}_{table_info['matched_count']}个匹配_表格{table_info['table_index']}"
            selected_tables[name] = table_info
    
   # 存储扩展后的表格（包含相邻表格）
    expanded_tables = {}
    max_index = len(tables) - 1
    
    # 首先添加原始选中的表格
    for name, info in selected_tables.items():
        table_index = info['table_index']
        expanded_tables[f"{name}_主表"] = info
        
        # 添加上一个表格（如果存在）
        if table_index > 0:
            prev_table = tables[table_index - 1]
            expanded_tables[f"{name}_上接表格{table_index-1}"] = {
                'table_index': table_index - 1,
                'df': prev_table.df,
                'matched_count': 0,  # 标记为相邻表格
                'matched_subjects': [],
                'shape': prev_table.df.shape,
                'is_adjacent': True
            }
        
        # 添加下一个表格（如果存在）
        if table_index < max_index:
            next_table = tables[table_index + 1]
            expanded_tables[f"{name}_下接表格{table_index+1}"] = {
                'table_index': table_index + 1,
                'df': next_table.df,
                'matched_count': 0,
                'matched_subjects': [],
                'shape': next_table.df.shape,
                'is_adjacent': True
            }
    
    # 打印结果
    print(f"\n总共选取了 {len(expanded_tables)} 个表格:")
    
    # 按类别和匹配数组织打印
    current_category = None
    for name, info in expanded_tables.items():
        # 从名称中提取类别
        category = name.split('_')[0]
        
        # 打印类别标题
        if category != current_category:
            print(f"\n=== {category} ===")
            current_category = category
        
        print(f"\n{name}:")
        print(f"- 表格大小: {info['shape']}")
        if info.get('is_adjacent', False):
            print("- 相邻表格")
        else:
            print(f"- 匹配数量: {info['matched_count']}")
            print(f"- 匹配的科目: {', '.join(info['matched_subjects'][:5])}...")
    
    # 将expanded_tables重组为按类别的DataFrame列表
    categorized_statements = {
        "资产负债表相关": [],
        "利润表相关": [],
        "现金流量表相关": []
    }
    
    # 整理表格
    for name, info in expanded_tables.items():
        # 从名称中提取类别
        category = name.split('_')[0]
        df = info['df']
        # 添加到对应类别的列表中
        categorized_statements[category].append(df)
    
    # 打印结果
    print(f"\n整理后的报表数量:")
    for category, df_list in categorized_statements.items():
        print(f"{category}: {len(df_list)}份")
    
    return categorized_statements

#%% Label Time to Tables
from concurrent.futures import ThreadPoolExecutor

def process_single_table(df, model_used, category):
   prompt = f"""
分析并标注财务表格的时间和科目信息。
表格内容:
{df.to_string()}
要求:
1. 判断是否为有效财务表格（包含会计科目和对应数值）
2. 标注时间（2023年度、2022年末等）和科目
3. 清理数值要求：
   - 删除括号、百分比等，保留纯数值
   - cleaned_value必须是有效数值，不能为空字符串
   - 如果单元格不包含数值，则不要包含在cells_to_label中
   - 对于标题、小计、合计等非数值行，不要包含在结果中
返回JSON格式要求：
{{
   "is_valid": true/false,  # 是否为有效财务表格
   "cells_to_label": [
       {{
           "row": 行索引,
           "col": 列索引,
           "cleaned_value": "数值字符串",  # 必须是有效数值
           "time_label": "时间标签",
           "subject_label": "科目标签"
       }}
   ]
}}"""

   try:
       response = client.chat.completions.create(
           model=model_used,
           messages=[
               {"role": "system", "content": "你是财务报表专家。"},
               {"role": "user", "content": prompt}
           ],
           temperature=0.1
       )
       print("Raw response:", response.choices[0].message.content)
       result = json.loads(clean_json_response(response.choices[0].message.content))
       
       if not result['is_valid']:
           return None
           
       labeled_df = df.copy()
       for cell in result['cells_to_label']:
           row, col = cell['row'], cell['col']
           new_value = ' '.join(str(x) for x in [
               cell['time_label'],
               cell['subject_label'],
               f"{float(cell['cleaned_value']):,}"
           ] if x)
           labeled_df.iloc[row, col] = new_value
           
       return labeled_df
       
   except Exception as e:
       print(f"处理{category}表格出错: {str(e)}")
       return None

def add_labels_using_gpt(model_used, main_tables):
   labeled_tables = {}
   
   with ThreadPoolExecutor(max_workers=1) as executor:
       for category, df_list in main_tables.items():
           print(f"处理{category}...")
           futures = [executor.submit(process_single_table, df, model_used, category) for df in df_list]
           
           # 过滤None结果
           labeled_dfs = [f.result() for f in futures if f.result() is not None]
           if labeled_dfs:
               labeled_tables[category] = labeled_dfs
               
           print(f"完成{category}, 保留{len(labeled_dfs)}张表")
           
   return labeled_tables

#%% Get Formulas
def get_all_formulas(model, matrix_list):
    prompt = f"""
对于以下financial matrices: {', '.join(matrix_list)}
Return ONLY a list of all unique required time points + subjects across all matrices. For example:
["2022年末应收账款", "2023年度净利润", "2023年末货币资金"]

Do NOT include duplicates. Each time point + subject combination should appear only once.
"""
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system", 
                "content": "You are a financial expert. Return ONLY the requested JSON format."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    
    try:
        formula_info = json.loads(response.choices[0].message.content)
        print(formula_info)
        return formula_info
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {str(e)}")
        print("原始内容:")
        print(response.choices[0].message.content)
        raise

#%% Extract Values
def extract_all_values(model_used, labeled_tables, formula_info):
    def format_table(df):
        return df.to_string(index=False)
        
    all_tables = {category: [format_table(df) for df in tables] 
                  for category, tables in labeled_tables.items()}
    
    # 将formula_info列表转换为格式化文本
    formatted_subjects = '\n'.join(f'- {item}' for item in formula_info)
    
    prompt = f"""提取财务报表数据。要求:
1. 处理特殊格式:括号数值转负数,百分比转小数
2. 识别不同表述:如"所有者权益"与"股东权益"等同
3. 相同科目多次出现时,根据上下文选择合理值 
4. 优先提取合计/总计行数据
提取科目:
{formatted_subjects}
财务报表数据:
{chr(10).join(f"{category}:{chr(10)}{chr(10).join(tables)}" for category, tables in all_tables.items())}
返回提取结果,JSON格式如下:
{{"科目名": 数值}}"""
    
    try:
        response = client.chat.completions.create(
            model=model_used,
            messages=[
                {"role": "system", "content": "你是专业的财务数据提取专家。请仅返回JSON格式的提取结果,不要有其他文字。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        values = json.loads(clean_json_response(response.choices[0].message.content))
        
        extracted = {k:v for k,v in values.items() if v is not None}
        print(f"\n提取结果 ({len(extracted)}/{len(formula_info)}):")
        print(json.dumps(extracted, indent=2, ensure_ascii=False))
           
        # 找出未提取的科目
        missing = [item for item in formula_info if item not in extracted]
        print(f"\n未提取科目 ({len(missing)}):")
        print(", ".join(missing))
    
        print(f"\n提取率: {len(extracted)/len(formula_info)*100:.1f}%")
    
        return extracted
       
    except Exception as e:
        print(f"提取错误: {str(e)}")
        return {}
        
#%% Calculate Matrix
def calculate_matrices(model, all_values, matrix_list):
    prompt = {
        "values": all_values,  # 直接使用数值字典
        "metrics": matrix_list
    }
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "金融指标计算专家,仅返回JSON格式结果"},
                {"role": "user", "content": """
根据提供的values计算metrics中的指标。
返回格式：{指标名称: 数值}

若计算错误，返回数值为错误码，规则如下：
- 缺少计算所需数据: -1001
- 除数为零: -1002
- 计算结果非法（如负数）: -1003
- 数据格式错误: -1004
- 其他计算错误: -1005

示例:
{
    "流动比率": 1.5,
    "资产周转率": -1001,  # 缺少数据
    "毛利率": -1002  # 除数为零
}
"""},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}
            ],
            temperature=0.1
        )
        
        result = json.loads(clean_json_response(response.choices[0].message.content))
        
        # 输出计算结果
        print("\n计算结果:")
        for metric, value in result.items():
            if value >= 0:
                print(f"{metric}: {value:.4f}")
            else:
                error_msgs = {
                    -1001: "缺少计算所需数据",
                    -1002: "除数为零",
                    -1003: "计算结果非法",
                    -1004: "数据格式错误",
                    -1005: "其他计算错误"
                }
                print(f"{metric}: {error_msgs.get(value, '未知错误')}")
                
        return result
        
    except Exception as e:
        print(f"计算错误: {str(e)}")
        return {}
    
#%% Restore Result
def create_results_df(calculation_results, company_code):
    """
    创建结果DataFrame，只包含matrix_list中的指标列
    
    Args:
        calculation_results: 包含计算结果的字典 {指标名: 数值}
        company_code: 公司代码
        
    Returns:
        DataFrame: 包含Stkcd和指标列的DataFrame
    """
    # 构建结果字典，以Stkcd为第一列
    data = {'Stkcd': company_code}
    
    # 检查是否有计算结果
    if calculation_results is None:
        # 如果计算完全失败，所有指标填-1001
        for matrix in matrix_list:
            data[matrix] = -1001
    else:
        # 对matrix_list中每个指标，填入计算结果或-1001
        for matrix in matrix_list:
            data[matrix] = calculation_results.get(matrix, -1001)
    
    # 创建DataFrame
    df = pd.DataFrame([data])
    df = df.round(2)
    return df

#%% Calculate Accuracy
def compare_dataframes(df1, ai_df, model):
    """比较两个DataFrame的值并返回准确率统计，包含不同容差范围的准确率"""
    if 'Stkcd' not in df1.columns or 'Stkcd' not in ai_df.columns:
        raise ValueError("两个DataFrame都必须包含Stkcd列")
    
    print(f"\n=== 比较结果 ({model}) ===")
    print(f"原始DataFrame形状: {df1.shape}")
    print(f"AI DataFrame形状: {ai_df.shape}")
    
    # 统一Stkcd列的数据类型为字符串
    df1_copy = df1.copy()
    ai_df_copy = ai_df.copy()
    
    df1_copy['Stkcd'] = df1_copy['Stkcd'].astype(str).str.zfill(6)
    ai_df_copy['Stkcd'] = ai_df_copy['Stkcd'].astype(str).str.zfill(6)
    
    # 合并DataFrame
    merged_df = pd.merge(ai_df_copy, df1_copy, on='Stkcd', how='left', suffixes=('_ai', '_orig'))
    
    # 存储比较结果
    accuracy_info = {
        'column_results': {},
        'column_accuracies': {},
        'tolerance_accuracies': {
            '1%': {},
            '5%': {},
            '10%': {}
        },
        'overall_results': {
            'total_matches': 0,
            'total_comparisons': 0,
            'overall_accuracy': 0,
            'tolerance_1': 0,
            'tolerance_5': 0,
            'tolerance_10': 0
        }
    }
    
    total_valid_comparisons = 0
    total_within_1 = 0
    total_within_5 = 0
    total_within_10 = 0
    
    # 遍历AI DataFrame的每一列（除了Stkcd）
    for col in ai_df.columns:
        if col == 'Stkcd':
            continue
            
        col_ai = f"{col}_ai"
        col_orig = f"{col}_orig"
        
        if col_orig not in merged_df.columns:
            print(f"警告：原始DataFrame中缺少列 {col}")
            accuracy_info['column_accuracies'][col] = 0
            accuracy_info['column_results'][col] = {
                'matches': 0, 
                'total': 0, 
                'accuracy': 0
            }
            continue
            
        # 计算相等的值的数量（跳过-101）
        valid_mask = (merged_df[col_orig] != -101) & (merged_df[col_ai] != -101)
        valid_rows = merged_df[valid_mask]
        
        if len(valid_rows) > 0:
            # 计算相对误差
            relative_error = abs(valid_rows[col_ai] - valid_rows[col_orig]) / abs(valid_rows[col_orig])
            
            # 计算各误差范围内的数量
            within_1 = (relative_error <= 0.01).sum()
            within_5 = (relative_error <= 0.05).sum()
            within_10 = (relative_error <= 0.10).sum()
            
            # 更新总计
            total_valid_comparisons += len(valid_rows)
            total_within_1 += within_1
            total_within_5 += within_5
            total_within_10 += within_10
            
            # 存储各列的容差准确率
            accuracy_info['tolerance_accuracies']['1%'][col] = within_1 / len(valid_rows)
            accuracy_info['tolerance_accuracies']['5%'][col] = within_5 / len(valid_rows)
            accuracy_info['tolerance_accuracies']['10%'][col] = within_10 / len(valid_rows)
        
        # 计算精确匹配
        matches = (valid_rows[col_orig] == valid_rows[col_ai]).sum()
        comparisons = len(valid_rows)
        
        # 计算这一列的正确率
        accuracy = matches / comparisons if comparisons > 0 else 0
        
        # 存储结果
        accuracy_info['column_accuracies'][col] = accuracy
        accuracy_info['column_results'][col] = {
            'matches': matches,
            'total': comparisons,
            'accuracy': accuracy
        }
        
        accuracy_info['overall_results']['total_matches'] += matches
        accuracy_info['overall_results']['total_comparisons'] += comparisons
    
    # 计算总体正确率
    if total_valid_comparisons > 0:
        accuracy_info['overall_results'].update({
            'overall_accuracy': accuracy_info['overall_results']['total_matches'] / 
                              accuracy_info['overall_results']['total_comparisons'],
            'tolerance_1': total_within_1 / total_valid_comparisons,
            'tolerance_5': total_within_5 / total_valid_comparisons,
            'tolerance_10': total_within_10 / total_valid_comparisons
        })
    
    # 打印结果
    print("\n各指标正确率:")
    for col, result in accuracy_info['column_results'].items():
        if result['total'] > 0:
            print(f"\n{col}:")
            print(f"精确匹配: {result['accuracy']*100:.2f}%")
            print(f"±1% 范围: {accuracy_info['tolerance_accuracies']['1%'].get(col, 0)*100:.2f}%")
            print(f"±5% 范围: {accuracy_info['tolerance_accuracies']['5%'].get(col, 0)*100:.2f}%")
            print(f"±10% 范围: {accuracy_info['tolerance_accuracies']['10%'].get(col, 0)*100:.2f}%")
    
    print(f"\n总体准确率:")
    print(f"精确匹配: {accuracy_info['overall_results']['overall_accuracy']*100:.2f}%")
    print(f"±1% 范围: {accuracy_info['overall_results']['tolerance_1']*100:.2f}%")
    print(f"±5% 范围: {accuracy_info['overall_results']['tolerance_5']*100:.2f}%")
    print(f"±10% 范围: {accuracy_info['overall_results']['tolerance_10']*100:.2f}%")
    
    return accuracy_info

#%% Save to Excel
def save_results_with_accuracy(model, results_df, accuracy_info, pdf_folder):
    """保存结果和准确率到Excel，包含不同容差范围的准确率"""
    try:
        # 处理文件名中的特殊字符
        safe_model = model.replace(':', '-')
        filename = f'results_{safe_model}.xlsx'
        filepath = os.path.join(pdf_folder, filename)
        
        print(f"准备保存到: {filepath}")
        print(f"检查results_df形状: {results_df.shape}")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 使用with语句自动处理关闭和保存
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # 保存results_df
            print("保存results sheet...")
            results_df.to_excel(writer, sheet_name='Results', index=False)
            
            # 创建准确率DataFrame
            metrics = list(accuracy_info['column_results'].keys())
            
            accuracy_data = []
            for metric in metrics:
                row = {
                    'Matrix': metric,
                    'Exact Match': accuracy_info['column_results'][metric]['accuracy'] * 100,
                    'Matches': accuracy_info['column_results'][metric]['matches'],
                    'Total': accuracy_info['column_results'][metric]['total']
                }
                
                # 添加不同容差范围的准确率
                for tolerance in ['1%', '5%', '10%']:
                    if tolerance in accuracy_info['tolerance_accuracies']:
                        row[f'Within ±{tolerance}'] = accuracy_info['tolerance_accuracies'][tolerance].get(metric, 0) * 100
                
                accuracy_data.append(row)
            
            # 创建DataFrame
            accuracy_df = pd.DataFrame(accuracy_data)
            print("创建accuracy_df成功，形状:", accuracy_df.shape)
            
            # 添加总体结果行
            overall_row = pd.DataFrame([{
                'Matrix': 'Overall',
                'Exact Match': accuracy_info['overall_results']['overall_accuracy'] * 100,
                'Within ±1%': accuracy_info['overall_results'].get('tolerance_1', 0) * 100,
                'Within ±5%': accuracy_info['overall_results'].get('tolerance_5', 0) * 100,
                'Within ±10%': accuracy_info['overall_results'].get('tolerance_10', 0) * 100,
                'Matches': accuracy_info['overall_results']['total_matches'],
                'Total': accuracy_info['overall_results']['total_comparisons']
            }])
            
            # 合并DataFrame
            accuracy_df = pd.concat([overall_row, accuracy_df], ignore_index=True)
            print("合并总体结果行成功，最终形状:", accuracy_df.shape)
            
            # 格式化准确率列
            percentage_columns = [col for col in accuracy_df.columns if 'Match' in col or 'Within' in col]
            for col in percentage_columns:
                accuracy_df[col] = accuracy_df[col].map('{:.2f}%'.format)
            
            # 保存准确率
            print("保存accuracy sheet...")
            accuracy_df.to_excel(writer, sheet_name='Accuracy', index=False)
            
        # 验证文件是否创建成功
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            print(f"文件成功保存到: {filepath}")
            print(f"文件大小: {os.path.getsize(filepath)} bytes")
            return filepath
        else:
            raise Exception("文件创建失败或大小为0")
        
    except Exception as e:
        print(f"保存文件时出错: {str(e)}")
        print(f"错误类型: {type(e)}")
        print(f"尝试保存位置: {filepath}")
        raise
#%%
if __name__ == "__main__":
    
    pdf_folder = r"D:\!MSAFA\！Course\3-GenAI\Project\Reports_2023"
    project_folder = r"D:\!MSAFA\！Course\3-GenAI\Project"
    matrix_list = matrix
    
    # 存储所有结果
    all_results = {}
    
    # 第一阶段:提取表格
    print("=== 提取财务报表 ===")
    tables_by_company = {}  # 存储每个公司的表格
    for _, row in df1.iterrows():
        stkcd = str(row['Stkcd']).zfill(6)
        print(f"\n处理公司 {stkcd}")
        
        # 检查PDF文件
        pattern = str(Path(pdf_folder) / f"{stkcd}*.pdf")
        matching_files = glob.glob(pattern)
        
        if matching_files:
            try:
                # 提取表格
                tables = camelot.read_pdf(matching_files[0], pages='all', flavor='stream', encoding='utf-8', suppress_stdout=True)
                main_tables = find_statements_by_subjects(tables)
                
                # 存储该公司的表格
                tables_by_company[stkcd] = main_tables
            except Exception as e:
                print(f"处理公司 {stkcd} 时出错: {str(e)}")
    
    # 第二阶段:模型处理
    # 设置模型
    model = "google:gemini-pro"#"openai:gpt-4o-mini"#'anthropic:claude-3-5-sonnet-latest'#'anthropic:claude-3-5-haiku-latest' #"mistral:our-favorite-model"#, "anthropic:claude-3-sonnet-20240229", "openai:gpt-4-turbo-preview", "google:gemini-pro"]

    print(f"\n=== 使用模型 {model} ===")
    results_df = pd.DataFrame(columns=['Stkcd'] + matrix_list)
    
    # 获取公式信息
    formula_info = get_all_formulas(model, matrix_list)
    
    # 处理每个公司
    for stkcd, main_tables in tables_by_company.items():
    #stkcd = '000552'
        print(f"\n处理公司 {stkcd}")
        
        try:
            # 使用当前模型打标
            labeled_tables = add_labels_using_gpt(model, main_tables)
            print(f"\n Successful labeled {stkcd}")
            
            # 提取值
            all_values = extract_all_values(model, labeled_tables, formula_info)
            print(f"\n Successful Extracted {stkcd}")
            
            # 计算指标
            results = calculate_matrices(model, all_values, matrix_list)
            print(f"\n Successful Calculated {stkcd}")
            
            # 创建结果DataFrame
            company_df = create_results_df(results, stkcd)
            
        except Exception as e:
            print(f"处理出错: {str(e)}")
            # 创建错误结果
            company_df = pd.DataFrame([[stkcd] + [-101]*len(matrix_list)], 
                                    columns=['Stkcd'] + matrix_list)
        
        # 合并结果
        results_df = pd.concat([results_df, company_df], ignore_index=True)
        
        # 计算准确率并保存结果
        accuracy_info = compare_dataframes(df1, results_df, model)
        save_results_with_accuracy(model, results_df, accuracy_info, project_folder)
    
        print(f"\n模型 {model} 公司{stkcd}处理完成")
   