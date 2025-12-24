"""
配置文件 - 量化策略参数设置

在这里集中管理所有可配置的参数
"""

import datetime

# ============================================================================
# 时间参数
# ============================================================================
START_DATE = datetime.date(2020, 1, 1)  # 回测开始日期
END_DATE = datetime.date(2025, 12, 15)  # 回测结束日期

# ============================================================================
# 投资周期参数
# ============================================================================
INVESTMENT_HORIZON = 'M'  # 'M': 月度调仓, 'W': 周度调仓, 'd': 日度调仓

# 每年的交易周期数
PERIODS_PER_YEAR = {
    'M': 12,   # 一年12个交易月
    'W': 52,   # 一年52个交易周
    'd': 252   # 一年252个交易日
}

# ============================================================================
# 数据参数
# ============================================================================
# 指数代码（中证1000）
INDEX_CODE = '000852.XSHG'

# 基准ETF代码（中证1000ETF）
BENCHMARK_ETF = '512100.XSHG'

# 因子类别（可选: 'risk', 'basics', 'growth', 'value', 'quality', 'momentum'等）
FACTOR_CATEGORIES = ['risk', 'basics']

# 训练数据年数（使用start_date之前多少年的数据）
TRAINING_YEARS = 3

# ============================================================================
# 选股参数
# ============================================================================
# 选股数量
N_STOCKS = 50

# 最小持仓权重
MIN_WEIGHT = 0.0

# ============================================================================
# 模型参数
# ============================================================================

# Ridge回归参数
RIDGE_PARAMS = {
    'alpha': 1.0
}

# Lasso回归参数
LASSO_PARAMS = {
    'alpha': 0.001,
    'max_iter': 10000
}

# ElasticNet参数
ELASTIC_PARAMS = {
    'alpha': 0.01,
    'l1_ratio': 0.5,
    'max_iter': 10000
}

# Random Forest参数
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1
}

# Gradient Boosting参数
GB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'subsample': 0.8,
    'random_state': 42
}

# 集成模型权重
ENSEMBLE_WEIGHTS = {
    'ridge': 0.15,
    'lasso': 0.15,
    'elastic': 0.15,
    'rf': 0.25,
    'gb': 0.30
}

# ============================================================================
# 回测参数
# ============================================================================
# 交易成本（双边）
TRANSACTION_COST = 0.001  # 0.1%

# 断点续跑文件名模板
SIMULATION_FILE_TEMPLATE = "L10_temp_optimized_{horizon}.pkl"

# ============================================================================
# 因子处理参数
# ============================================================================
# Winsorization分位数
WINSORIZE_QUANTILES = [0.01, 0.99]

# 极端值阈值
EXTREME_VALUE_THRESHOLD = 101

# 标准差最小阈值
MIN_STD = 0.000001

# 低基数因子阈值
LOW_CARDINALITY_THRESHOLD = 20

# ============================================================================
# 辅助函数
# ============================================================================

def get_simulation_file():
    """获取当前投资周期对应的模拟文件名"""
    return SIMULATION_FILE_TEMPLATE.format(horizon=INVESTMENT_HORIZON.lower())


def get_number_of_periods_per_year():
    """获取当前投资周期对应的年度周期数"""
    return PERIODS_PER_YEAR[INVESTMENT_HORIZON]


def validate_config():
    """验证配置参数的有效性"""
    errors = []
    
    # 验证投资周期
    if INVESTMENT_HORIZON not in PERIODS_PER_YEAR:
        errors.append(f"无效的投资周期: {INVESTMENT_HORIZON}，必须是 'M', 'W', 或 'd'")
    
    # 验证日期
    if START_DATE >= END_DATE:
        errors.append(f"开始日期 {START_DATE} 必须早于结束日期 {END_DATE}")
    
    # 验证选股数量
    if N_STOCKS <= 0:
        errors.append(f"选股数量 {N_STOCKS} 必须大于0")
    
    # 验证集成权重
    weight_sum = sum(ENSEMBLE_WEIGHTS.values())
    if abs(weight_sum - 1.0) > 0.001:
        errors.append(f"集成权重之和 {weight_sum} 必须等于1.0")
    
    # 验证交易成本
    if TRANSACTION_COST < 0 or TRANSACTION_COST > 0.1:
        errors.append(f"交易成本 {TRANSACTION_COST} 应该在0到0.1之间")
    
    if errors:
        print("配置验证失败:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True


def print_config():
    """打印当前配置"""
    print("=" * 70)
    print("当前配置参数")
    print("=" * 70)
    print(f"回测时间范围: {START_DATE} 至 {END_DATE}")
    print(f"投资周期: {INVESTMENT_HORIZON} (每年 {get_number_of_periods_per_year()} 个周期)")
    print(f"训练数据: 使用开始日期前 {TRAINING_YEARS} 年的数据")
    print(f"指数代码: {INDEX_CODE}")
    print(f"基准ETF: {BENCHMARK_ETF}")
    print(f"因子类别: {FACTOR_CATEGORIES}")
    print(f"选股数量: {N_STOCKS}")
    print(f"交易成本: {TRANSACTION_COST*100:.2f}%")
    print(f"模拟文件: {get_simulation_file()}")
    print("\n集成模型权重:")
    for model, weight in ENSEMBLE_WEIGHTS.items():
        print(f"  - {model}: {weight*100:.1f}%")
    print("=" * 70)


if __name__ == '__main__':
    # 验证并打印配置
    if validate_config():
        print_config()
        print("\n✓ 配置验证通过")
    else:
        print("\n✗ 配置验证失败")
