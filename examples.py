"""
使用示例和测试脚本

本脚本演示如何使用优化后的量化策略
注意：实际运行需要聚宽（JoinQuant）环境
"""

import datetime

# ============================================================================
# 示例 1: 使用默认配置运行策略
# ============================================================================

def example_1_default_config():
    """使用默认配置运行策略"""
    print("=" * 70)
    print("示例 1: 使用默认配置运行策略")
    print("=" * 70)
    
    # 直接运行 strategy.py 即可
    # python strategy.py
    
    print("""
在聚宽环境中运行:
1. 复制 strategy.py 的内容到聚宽研究环境
2. 直接运行
3. 等待训练和回测完成
4. 查看信息比率等性能指标
    """)


# ============================================================================
# 示例 2: 自定义配置
# ============================================================================

def example_2_custom_config():
    """使用自定义配置"""
    print("=" * 70)
    print("示例 2: 使用自定义配置")
    print("=" * 70)
    
    print("""
修改 config.py 中的参数:

# 改变投资周期为周度
INVESTMENT_HORIZON = 'W'

# 改变选股数量
N_STOCKS = 30

# 调整模型权重（偏重机器学习模型）
ENSEMBLE_WEIGHTS = {
    'ridge': 0.10,
    'lasso': 0.10,
    'elastic': 0.10,
    'rf': 0.30,
    'gb': 0.40
}

然后运行 strategy_v2.py
    """)


# ============================================================================
# 示例 3: 重新训练模型
# ============================================================================

def example_3_retrain():
    """重新训练模型"""
    print("=" * 70)
    print("示例 3: 重新训练模型")
    print("=" * 70)
    
    print("""
当需要重新训练模型时（例如修改了模型参数或因子）:

步骤 1: 清理旧的模拟文件
    python cleanup.py M
    
    或手动删除:
    import os
    os.remove('L10_temp_optimized_m.pkl')

步骤 2: 重新运行策略
    python strategy_v2.py
    
    策略会从头开始训练模型并进行回测
    """)


# ============================================================================
# 示例 4: 切换投资周期
# ============================================================================

def example_4_switch_horizon():
    """切换投资周期"""
    print("=" * 70)
    print("示例 4: 切换投资周期（月度 -> 周度）")
    print("=" * 70)
    
    print("""
从月度调仓切换到周度调仓:

步骤 1: 修改 config.py
    INVESTMENT_HORIZON = 'W'  # 改为 'W'

步骤 2: 清理月度模拟文件（可选）
    python cleanup.py M

步骤 3: 运行策略
    python strategy_v2.py
    
    会自动创建新的周度模拟文件 L10_temp_optimized_w.pkl
    """)


# ============================================================================
# 示例 5: 测试不同的因子组合
# ============================================================================

def example_5_factor_selection():
    """测试不同的因子组合"""
    print("=" * 70)
    print("示例 5: 测试不同的因子组合")
    print("=" * 70)
    
    print("""
尝试不同的因子类别组合:

在 config.py 中修改:

# 选项 1: 只使用 risk 因子
FACTOR_CATEGORIES = ['risk']

# 选项 2: 添加更多因子类别
FACTOR_CATEGORIES = ['risk', 'basics', 'growth', 'value']

# 选项 3: 使用所有可用因子
FACTOR_CATEGORIES = ['risk', 'basics', 'growth', 'value', 'quality', 'momentum']

注意: 
- 添加更多因子可能提高预测能力，但也会增加计算时间
- 需要清理旧的模拟文件后重新训练
- 可以比较不同因子组合的信息比率
    """)


# ============================================================================
# 示例 6: 优化模型参数
# ============================================================================

def example_6_tune_params():
    """优化模型参数"""
    print("=" * 70)
    print("示例 6: 优化模型参数")
    print("=" * 70)
    
    print("""
调整模型超参数以获得更好的性能:

在 config.py 中修改:

# Random Forest 参数调优
RF_PARAMS = {
    'n_estimators': 200,      # 增加树的数量
    'max_depth': 15,          # 增加树的深度
    'min_samples_split': 10,  # 减小分裂样本数
    'min_samples_leaf': 5,    # 减小叶子节点样本数
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1
}

# Gradient Boosting 参数调优
GB_PARAMS = {
    'n_estimators': 150,
    'max_depth': 6,
    'learning_rate': 0.05,    # 降低学习率
    'min_samples_split': 15,
    'min_samples_leaf': 8,
    'subsample': 0.9,
    'random_state': 42
}

建议:
1. 使用网格搜索或随机搜索找到最优参数
2. 在验证集上评估参数效果
3. 注意过拟合风险
    """)


# ============================================================================
# 性能分析指南
# ============================================================================

def guide_performance_analysis():
    """性能分析指南"""
    print("=" * 70)
    print("性能分析指南")
    print("=" * 70)
    
    print("""
关键性能指标:

1. 信息比率 (Information Ratio)
   - 最重要的指标
   - 公式: IR = 超额收益均值 / 超额收益标准差
   - 目标: 最大化该值
   - 一般标准:
     * IR > 0.5: 良好
     * IR > 1.0: 优秀
     * IR > 1.5: 卓越

2. 夏普比率 (Sharpe Ratio)
   - 风险调整后的收益
   - 公式: SR = (收益率 - 无风险利率) / 收益率标准差
   - 目标: 越高越好

3. 年化收益率
   - 策略的年化回报
   - 对比基准ETF的收益

4. 最大回撤
   - 从峰值到谷底的最大跌幅
   - 反映风险控制能力

5. 胜率
   - 盈利周期占比
   - 反映策略稳定性

在策略执行完成后，可以添加以下代码进行详细分析:

# 详细性能分析
print("\\n详细性能分析:")
print(f"策略总收益: {(wealth_process.iloc[-1] - 1) * 100:.2f}%")
print(f"基准总收益: {(combined_df['基准ETF'].iloc[-1] - 1) * 100:.2f}%")
print(f"超额收益: {(wealth_process.iloc[-1] - combined_df['基准ETF'].iloc[-1]) * 100:.2f}%")

# 计算年化收益
years = (end_date - start_date).days / 365.25
annualized_return = (wealth_process.iloc[-1] ** (1/years) - 1) * 100
print(f"策略年化收益: {annualized_return:.2f}%")

# 计算最大回撤
cummax = wealth_process.cummax()
drawdown = (wealth_process - cummax) / cummax
max_drawdown = drawdown.min() * 100
print(f"最大回撤: {max_drawdown:.2f}%")
    """)


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数：显示所有示例"""
    print("\n")
    print("*" * 70)
    print("量化策略使用示例和指南")
    print("*" * 70)
    print("\n")
    
    example_1_default_config()
    print("\n")
    
    example_2_custom_config()
    print("\n")
    
    example_3_retrain()
    print("\n")
    
    example_4_switch_horizon()
    print("\n")
    
    example_5_factor_selection()
    print("\n")
    
    example_6_tune_params()
    print("\n")
    
    guide_performance_analysis()
    print("\n")
    
    print("*" * 70)
    print("注意事项:")
    print("*" * 70)
    print("""
1. 所有示例都需要在聚宽（JoinQuant）环境中运行
2. 修改配置后记得清理旧的模拟文件
3. 训练过程可能需要较长时间，请耐心等待
4. 建议先在小数据集上测试，确认无误后再运行完整回测
5. 定期保存重要的回测结果和性能指标
6. 注意模型过拟合问题，使用交叉验证
7. 关注信息比率作为主要优化目标
    """)
    print("*" * 70)


if __name__ == '__main__':
    main()
