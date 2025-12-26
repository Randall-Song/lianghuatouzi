#!/usr/bin/env python3
"""
验证脚本 - 检查策略代码的关键功能
此脚本用于在非聚宽环境中验证代码结构的正确性
"""

import sys
import ast
import inspect

def check_strategy_structure():
    """检查strategy.py的结构是否正确"""
    
    print("="*80)
    print("策略代码结构验证")
    print("="*80)
    
    # 读取策略文件
    with open('strategy.py', 'r', encoding='utf-8') as f:
        code = f.read()
    
    # 解析AST
    try:
        tree = ast.parse(code)
        print("✓ Python语法检查通过")
    except SyntaxError as e:
        print(f"✗ 语法错误: {e}")
        return False
    
    # 检查必需的函数
    required_functions = [
        'get_st_or_paused_stock_set',
        'cal_vwap_ret_series',
        'cal_portfolio_vwap_ret',
        'get_previous_trade_date',
        'get_next_trade_date',
        'get_buy_dates',
        'normalize_series',
        'get_my_factors',
        'train_my_model',
        'load_or_train_model',
        'cal_portfolio_weight_series',
        'simulate_wealth_process',
        'evaluate_performance'
    ]
    
    found_functions = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            found_functions.add(node.name)
    
    print("\n检查必需函数:")
    all_found = True
    for func in required_functions:
        if func in found_functions:
            print(f"  ✓ {func}")
        else:
            print(f"  ✗ {func} 缺失")
            all_found = False
    
    # 检查cal_portfolio_weight_series的签名
    print("\n检查关键函数签名:")
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'cal_portfolio_weight_series':
            args = [arg.arg for arg in node.args.args]
            expected_args = ['decision_date', 'old_portfolio_weight_series']
            if args == expected_args:
                print(f"  ✓ cal_portfolio_weight_series 签名正确: {args}")
            else:
                print(f"  ✗ cal_portfolio_weight_series 签名错误")
                print(f"    期望: {expected_args}")
                print(f"    实际: {args}")
                all_found = False
    
    # 检查必需的导入
    print("\n检查必需的导入:")
    required_imports = [
        'pandas',
        'numpy',
        'tqdm',
        'datetime',
        'pickle',
        'os',
        'sklearn.linear_model',
        'sklearn.ensemble'
    ]
    
    found_imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found_imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            found_imports.add(node.module)
    
    for imp in required_imports:
        if any(imp in found for found in found_imports):
            print(f"  ✓ {imp}")
        else:
            print(f"  ⚠ {imp} (可能在聚宽环境中使用)")
    
    # 检查配置变量
    print("\n检查配置变量:")
    required_vars = [
        'start_date',
        'end_date',
        'investment_horizon',
        'number_of_periods_per_year',
        'simulation_file',
        'model_file'
    ]
    
    for var in required_vars:
        if var in code:
            print(f"  ✓ {var}")
        else:
            print(f"  ✗ {var} 缺失")
            all_found = False
    
    # 检查GradientBoostingRegressor的使用
    print("\n检查模型类型:")
    if 'GradientBoostingRegressor' in code:
        print("  ✓ 使用 GradientBoostingRegressor (改进模型)")
    else:
        print("  ✗ 未找到 GradientBoostingRegressor")
    
    if 'Ridge' in code and 'from sklearn.linear_model import Ridge' in code:
        print("  ⚠ 仍然导入了 Ridge (但可能未使用)")
    
    # 检查训练数据时间范围
    print("\n检查训练数据配置:")
    if 'timedelta(365*4)' in code or 'timedelta(days=365*4)' in code:
        print("  ✓ 训练数据使用4年历史数据")
    elif 'timedelta(365*3)' in code or 'timedelta(days=365*3)' in code:
        print("  ⚠ 训练数据使用3年历史数据（建议使用4年）")
    
    # 检查股票选择数量
    print("\n检查投资组合配置:")
    if 'n_stocks = 40' in code or 'n_stocks=40' in code:
        print("  ✓ 选股数量设置为40只")
    elif '.nlargest(40)' in code:
        print("  ✓ 选股数量设置为40只")
    elif '.nlargest(50)' in code:
        print("  ⚠ 选股数量为50只（建议改为40只）")
    
    # 检查权重方法
    if 'predicted_weights.sum()' in code:
        print("  ✓ 使用预测值加权方法")
    elif '1/len(filtered_assets)' in code:
        print("  ⚠ 使用等权方法（建议改为预测值加权）")
    
    print("\n" + "="*80)
    if all_found:
        print("✓ 所有检查通过!")
        print("="*80)
        return True
    else:
        print("⚠ 部分检查未通过，请检查上述标记为 ✗ 的项")
        print("="*80)
        return False

def check_file_structure():
    """检查文件结构"""
    import os
    
    print("\n" + "="*80)
    print("文件结构检查")
    print("="*80)
    
    required_files = {
        'strategy.py': '主策略文件',
        'README.md': '使用说明文档',
        'IMPROVEMENTS.md': '改进说明文档',
        '.gitignore': 'Git忽略配置'
    }
    
    for filename, description in required_files.items():
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"✓ {filename:20s} ({size:,} bytes) - {description}")
        else:
            print(f"✗ {filename:20s} 缺失 - {description}")
    
    # 检查.gitignore内容
    if os.path.exists('.gitignore'):
        with open('.gitignore', 'r') as f:
            content = f.read()
        if '*.pkl' in content:
            print("  ✓ .gitignore 正确配置忽略 .pkl 文件")
        else:
            print("  ⚠ .gitignore 可能未正确配置")
    
    print("="*80)

if __name__ == '__main__':
    success = check_strategy_structure()
    check_file_structure()
    
    if success:
        print("\n✓ 策略代码验证通过！")
        print("\n使用说明:")
        print("1. 在聚宽平台上运行 strategy.py")
        print("2. 首次运行会自动训练模型并保存到 my_model.pkl")
        print("3. 后续运行会自动加载已训练的模型")
        print("4. 如需重新训练模型，删除 my_model.pkl 和 L10_temp_fixed_m_basicsrisk.pkl")
        sys.exit(0)
    else:
        print("\n⚠ 策略代码存在问题，请修复后再运行")
        sys.exit(1)
