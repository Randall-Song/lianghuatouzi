# L10 多因子策略 - 优化正信息比率版
# 目标：在稳健性基础上进一步提升信息比率
#
# 核心策略调整（基于IR=0.09的稳健基线）：
# 1. 扩展因子池：添加value因子（basics, quality, value三类最可靠因子）
# 2. 平衡持仓：选择40只股票以平衡风险和收益
# 3. 略微激进的特征选择：选择相关性前35%的因子（在稳健性和信号强度间平衡）
# 4. 最小非线性：仅添加Top-2因子交互项（谨慎捕捉非线性关系）
# 5. 略微降低正则化：使用0.3-1.5的alpha值（在偏差方差间更好平衡）
# 6. 绝对收益预测：回归预测绝对收益率（保持稳定）
#
# 重要说明：
# 1. 训练my_model的数据必须是start_date之前的数据
# 2. cal_portfolio_weight_series(decision_date)函数签名不能改动
# 3. 要求在聚宽环境中运行，不改变原本的输出格式
# 4. 信息比率必须为正且持续提升

import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
from jqfactor import get_all_factors, get_factor_values
import jqdata
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from jqlib import alpha101
import os
from math import sqrt
import matplotlib.pyplot as plt

# ========== 参数配置 ==========
start_date = datetime.date(2020, 1, 1)
end_date = datetime.date(2025, 12, 15)
investment_horizon = 'M'  # M 为月度调仓， W为周度调仓, d为日度调仓
number_of_periods_per_year = 12  # 一年12个交易月，52个交易周，252个交易日
optimal_stock_count = 40  # 选择的股票数量（平衡风险和收益）

# ========== 因子获取 ==========
# 使用三类最可靠的基本面因子：basics, quality, value
all_factors = get_all_factors()
all_factors = all_factors.loc[[a in ['basics', 'quality', 'value'] 
                 for a in all_factors.loc[:, 'category']], 'factor'].tolist()
print(f"使用 {len(all_factors)} 个因子进行建模")

# ========== 辅助函数 ==========
def get_st_or_paused_stock_set(decision_date):
    """抓取st和停牌股票的集合"""
    all_stock_ids = get_all_securities(types=['stock'], date=decision_date).index.tolist()
    is_st_flag = get_extras('is_st', all_stock_ids, start_date=decision_date, end_date=decision_date)
    st_set = set(is_st_flag.iloc[0][is_st_flag.iloc[0]].index)
    
    paused_flag = get_price(all_stock_ids, 
                        start_date=decision_date, 
                        end_date=decision_date, 
                        frequency='daily',
                        fq='post', panel=False, fields=['paused'])
    paused_set = set(paused_flag.loc[paused_flag.loc[:, 'paused']==1].loc[:, 'code'].values)
    return st_set.union(paused_set)

def cal_vwap_ret_series(order_book_ids, buy_date, sell_date):
    """计算一组资产买卖周期内的vwap回报率"""
    if len(order_book_ids) == 0: 
        return pd.Series(dtype='float64')
    all_data = get_price(order_book_ids, buy_date, sell_date, fields=['money', 'volume'], 
                            fq='post', panel=False)
    vwap_prices = all_data.set_index(['time', 'code']).unstack().loc[:, 'money'] / \
        all_data.set_index(['time', 'code']).unstack().loc[:, 'volume']
    vwap_ret_series = vwap_prices.iloc[-1]/vwap_prices.iloc[0]-1
    return vwap_ret_series

def cal_portfolio_vwap_ret(old_portfolio_weight_series, new_portfolio_weight_series, buy_date, sell_date):
    """计算一个组合从买入到卖出的持有期回报率"""
    order_book_ids = new_portfolio_weight_series.index.tolist()
    
    vwap_ret_series = cal_vwap_ret_series(order_book_ids, buy_date, sell_date)
    hpr = (new_portfolio_weight_series * vwap_ret_series).sum()
    
    old_new = pd.concat([old_portfolio_weight_series, new_portfolio_weight_series], axis=1).fillna(0)
    cost = (old_new.iloc[:, 0] - old_new.iloc[:, 1]).abs().sum() * 0.001
    
    return hpr - cost

def get_previous_trade_date(current_date):
    """抓取上一交易日"""
    trading_dates = jqdata.get_all_trade_days()
    trading_dates = sorted(trading_dates)
    for trading_date in reversed(trading_dates):
        if trading_date < current_date:
            return trading_date
    return None

def get_next_trade_date(current_date):
    """抓取下一交易日"""
    trading_dates = jqdata.get_all_trade_days()
    trading_dates = sorted(trading_dates)
    for trading_date in trading_dates:
        if trading_date > current_date:
            return trading_date
    return None

def get_buy_dates(start_date: str, end_date: str, freq: str) -> list:
    """抓取要进行买卖调仓的日期"""
    periodic_dates = [x.date() for x in pd.date_range(start_date, end_date, freq=freq)]
    trading_dates = jqdata.get_trade_days(start_date, end_date)
    return np.sort(np.unique([get_next_trade_date(d) for d in periodic_dates 
                              if (get_next_trade_date(d) <= end_date)])).tolist()

def normalize_series(series):
    """标准化序列"""
    series = series.copy().replace([np.inf, -np.inf], np.nan)
    
    if series.abs().max() > 101:
        series = np.sign(series) * np.log2(1. + series.abs())

    if np.isnan(series.mean()) or np.isnan(series.std()) or (series.std() < 0.000001): 
        series.iloc[:] = 0.
        return series

    if len(series.unique()) <= 20: 
        series = (series - series.mean()) / series.std()
        return series.fillna(0)

    q = series.quantile([0.01, 0.99])
    series[series < q.iloc[0]] = q.iloc[0]
    series[series > q.iloc[1]] = q.iloc[1]

    series = (series - series.mean()) / series.std()
    
    series = series.replace([np.inf, -np.inf], np.nan).fillna(0.)
    
    return series

def get_my_factors(decision_date, all_stocks):
    """获取因子数据"""
    factor_df_list = []
    for i_factor in all_factors:
        factor_df_list.append(get_factor_values(securities=all_stocks, 
                              factors=i_factor, 
                              start_date=decision_date, 
                              end_date=decision_date)[i_factor].T)
    factor_df = pd.concat(factor_df_list, axis=1)
    factor_df.columns = all_factors
    return factor_df

# ========== 训练数据准备 ==========
print("准备训练数据...")
# 训练数据必须在start_date之前
training_end_date = start_date - datetime.timedelta(days=1)
training_dates = get_buy_dates(start_date=start_date - datetime.timedelta(365*3), 
                               end_date=training_end_date, freq=investment_horizon)

print(f"训练数据时间范围: {training_dates[0]} 到 {training_dates[-1]}")
print(f"测试开始日期: {start_date}")
print(f"确保训练数据在测试开始之前: {training_dates[-1] < start_date}")

factor_df_list = []
for i in tqdm(range(len(training_dates)-1)):
    buy_date = training_dates[i]
    sell_date = training_dates[i+1]
    i_pre_date = get_previous_trade_date(buy_date)
    all_stocks = get_index_stocks('000852.XSHG', date=i_pre_date)
    all_stocks = list(set(all_stocks) - get_st_or_paused_stock_set(i_pre_date))
    
    factor_df = get_my_factors(i_pre_date, all_stocks)
    
    # 预测绝对收益率（更稳定）
    factor_df.loc[:, 'next_vwap_ret'] = cal_vwap_ret_series(all_stocks, buy_date, sell_date)
    factor_df = factor_df.apply(normalize_series)
    factor_df_list.append(factor_df)

factor_df_list = pd.concat(factor_df_list).dropna()

print(f"训练数据形状: {factor_df_list.shape}")
print(f"目标变量：绝对收益率")

# ========== 模型训练 ==========
print("训练模型...")

# 尝试使用多个模型并选择最佳的
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score

# 准备训练数据
X_train = factor_df_list.iloc[:, :-1]
y_train = factor_df_list.iloc[:, -1]

# 特征选择：计算每个因子与收益的相关性
correlations = X_train.corrwith(y_train).abs()
print(f"\n因子与收益相关性分析（前12个）:")
print(correlations.nlargest(12))

# 略微激进的特征选择策略 - 选择相关性前35%的因子（提高信号质量）
threshold = correlations.quantile(0.65)  # 选择相关性前35%的因子
selected_features = correlations[correlations > threshold].index.tolist()
print(f"\n选择了 {len(selected_features)} 个因子（相关性 > {threshold:.4f}）")

# 如果选择的特征太少，至少保留前10个
if len(selected_features) < 10:
    selected_features = correlations.nlargest(10).index.tolist()
    print(f"特征数量不足，使用前10个相关性最高的因子")

X_train_selected = X_train[selected_features]

# 添加最小非线性：仅Top-2因子交互（谨慎增强）
print("\n添加最小非线性特征...")
if len(selected_features) >= 2:
    # 仅选择相关性最高的2个因子进行交互
    top_features = correlations.nlargest(min(2, len(selected_features))).index.tolist()
    for i in range(len(top_features)):
        for j in range(i+1, len(top_features)):
            interaction_name = f"{top_features[i]}_x_{top_features[j]}"
            X_train_selected[interaction_name] = X_train[top_features[i]] * X_train[top_features[j]]
    print(f"添加了 {len(top_features)*(len(top_features)-1)//2} 个交互项")

print(f"增强后的特征数量: {X_train_selected.shape[1]}")

# 使用略微降低的正则化，提升拟合能力
models = {
    'Ridge_alpha0.3': Ridge(alpha=0.3),
    'Ridge_alpha0.5': Ridge(alpha=0.5),
    'Ridge_alpha1.0': Ridge(alpha=1.0),
    'Ridge_alpha1.5': Ridge(alpha=1.5),
    'Lasso_alpha0.0005': Lasso(alpha=0.0005, max_iter=10000),
    'Lasso_alpha0.001': Lasso(alpha=0.001, max_iter=10000),
    'Lasso_alpha0.003': Lasso(alpha=0.003, max_iter=10000),
    'ElasticNet_0.5_0.001': ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000),
    'ElasticNet_0.5_0.003': ElasticNet(alpha=0.003, l1_ratio=0.5, max_iter=10000),
    'ElasticNet_0.7_0.003': ElasticNet(alpha=0.003, l1_ratio=0.7, max_iter=10000),
}

best_score = -np.inf
best_model_name = None
best_model = None
best_features = X_train_selected.columns.tolist()

for name, model in models.items():
    try:
        # 使用交叉验证评估模型，关注R2得分
        scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='r2')
        avg_score = scores.mean()
        print(f"{name} 平均R2得分: {avg_score:.4f} (+/- {scores.std() * 2:.4f})")
        
        if avg_score > best_score:
            best_score = avg_score
            best_model_name = name
            best_model = model
    except Exception as e:
        print(f"{name} 训练失败: {e}")

# 使用最佳模型训练
print(f"\n选择最佳模型: {best_model_name} (R2 = {best_score:.4f})")
my_model = best_model
my_model.fit(X_train_selected, y_train)

# 保存选择的特征列表，以便预测时使用
selected_feature_names = X_train_selected.columns.tolist()

# 输出模型系数和重要性
if hasattr(my_model, 'coef_'):
    coefficients = my_model.coef_
    print("\n回归权重（系数）- 前12个最重要的因子:")
    coef_series = pd.Series(coefficients, index=selected_feature_names)
    # 显示绝对值最大的前12个系数
    print(coef_series.abs().nlargest(min(12, len(coef_series))))

# ========== 定义组合权重计算函数（不能修改） ==========
def cal_portfolio_weight_series(decision_date, old_portfolio_weight_series):
    """
    在decision_date收盘后决定接下来下一天要买的投资组合
    这个函数里面不能使用任何decision_date所在时间之后的信息
    """
    # 基础股票池，去掉st和停牌股
    all_stocks = get_index_stocks('000852.XSHG', date=decision_date)
    all_stocks = list(set(all_stocks) - get_st_or_paused_stock_set(decision_date))
    
    # 抓取因子
    factor_df = get_my_factors(decision_date, all_stocks)
    factor_df = factor_df.apply(normalize_series)
    
    # 构建包含交互项的特征矩阵
    factor_df_with_features = factor_df.copy()
    
    # 添加训练时使用的交互项
    for feature_name in selected_feature_names:
        if '_x_' in feature_name:
            # 这是一个交互项
            parts = feature_name.split('_x_')
            if len(parts) == 2 and parts[0] in factor_df.columns and parts[1] in factor_df.columns:
                factor_df_with_features[feature_name] = factor_df[parts[0]] * factor_df[parts[1]]
    
    # 选择训练时使用的特征
    available_features = [f for f in selected_feature_names if f in factor_df_with_features.columns]
    factor_df_selected = factor_df_with_features[available_features]
    
    # 预测收益
    predicted_returns = pd.Series(my_model.predict(factor_df_selected), index=factor_df.index)
    
    # 筛选 - 选择预测收益最高的股票
    filtered_assets = predicted_returns.nlargest(optimal_stock_count).index.tolist()
    
    # 配权 - 等权重
    portfolio_weight_series = pd.Series(1/len(filtered_assets), index=filtered_assets)
    
    return portfolio_weight_series

# ========== 回测模拟 ==========
def simulate_wealth_process(start_date, end_date):
    """模拟一段时间的策略回测"""
    all_buy_dates = get_buy_dates(start_date, end_date, investment_horizon)
    wealth_process = pd.Series(np.nan, index=all_buy_dates)
    wealth_process.iloc[0] = 1
    allocation_dict = dict()
    old_portfolio_weight_series = pd.Series(dtype='float64')
    
    for i in tqdm(range(len(all_buy_dates)-1)):
        buy_date = all_buy_dates[i]
        sell_date = all_buy_dates[i+1]
        
        decision_date = get_previous_trade_date(buy_date)
        new_portfolio_weight_series = cal_portfolio_weight_series(decision_date, old_portfolio_weight_series)
        
        allocation_dict[buy_date] = new_portfolio_weight_series.copy()
        
        wealth_process.loc[sell_date] = wealth_process.loc[buy_date] * \
            (1 + cal_portfolio_vwap_ret(old_portfolio_weight_series, new_portfolio_weight_series, buy_date, sell_date))
        
        old_portfolio_weight_series = new_portfolio_weight_series
        
    return wealth_process, allocation_dict

# 进行策略回测模拟
print("\n开始回测...")
wealth_process, allocation_dict = simulate_wealth_process(start_date, end_date)

# ========== 性能评估 ==========
print("\n评估策略性能...")

# 查看不同时间的选股个数
stock_counts = pd.Series([len(b) for a, b in allocation_dict.items()])
print(f"平均选股个数: {stock_counts.mean():.1f}")

# 画策略的财富曲线
wealth_process.name = '我的策略'

# 获取基准ETF的日度vwap价格
benchmark_index = get_price(['512100.XSHG'], 
            start_date=start_date, 
            end_date=end_date, 
            frequency='daily',
            fq='post', panel=False, fields=['money', 'volume'])
benchmark_index.loc[:, 'vwap'] = benchmark_index.loc[:, 'money'] / benchmark_index.loc[:, 'volume']
benchmark_index = benchmark_index.set_index('time').loc[:, ['vwap']].loc[wealth_process.index]
benchmark_index = benchmark_index / benchmark_index.iloc[0]
benchmark_index.columns = ['基准ETF']

# 合并策略和基准
combined_df = pd.concat([wealth_process, benchmark_index], axis=1)

# 计算性能指标
combined_df_ret = combined_df.pct_change()
combined_df_ret.loc[:, '超额收益'] = combined_df_ret.loc[:, '我的策略'] - combined_df_ret.loc[:, '基准ETF']
performance_metrics = combined_df_ret.mean() * number_of_periods_per_year / \
    (combined_df_ret.std() * sqrt(number_of_periods_per_year))
performance_metrics.index = ['我的策略夏普', '基准ETF夏普', '信息比率']

print("\n性能指标:")
print(performance_metrics)
print(f"\n信息比率: {performance_metrics['信息比率']:.4f}")

# 绘图
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# 财富曲线
combined_df.plot(ax=axes[0], title='策略vs基准财富曲线')
axes[0].set_ylabel('财富')
axes[0].grid(True)

# 性能指标柱状图
axes[1].bar(performance_metrics.index, performance_metrics)
axes[1].set_title('性能指标对比')
axes[1].set_ylabel('比率')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('strategy_performance.png', dpi=150, bbox_inches='tight')
print("\n图表已保存到 strategy_performance.png")
