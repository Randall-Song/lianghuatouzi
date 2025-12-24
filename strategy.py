"""
多因子量化策略 - 优化版
目标：设计和训练my_model，最大化信息比率

要求：
1. cal_portfolio_weight_series(decision_date, old_portfolio_weight_series)函数不能改动
2. 训练my_model的数据必须是start_date之前的数据
3. 重新设计模型需要把本地支持断点续跑的simulation_file删除
4. investment_horizon可以选W或者M，要求在聚宽环境中能运行
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
from jqfactor import get_all_factors, get_factor_values
import jqdata
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from jqlib import alpha101
import pickle
import os
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from math import sqrt

# ============================================================================
# 配置参数
# ============================================================================
start_date = datetime.date(2020, 1, 1)
end_date = datetime.date(2025, 12, 15)
investment_horizon = 'M'  # M 为月度调仓, W为周度调仓, d为日度调仓
number_of_periods_per_year = {'M': 12, 'W': 52, 'd': 252}[investment_horizon]
simulation_file = f"L10_temp_optimized_{investment_horizon.lower()}.pkl"

# 获取因子列表 - 使用risk和basics类别的因子
all_factors = get_all_factors()
all_factors = all_factors.loc[[a in ['risk', 'basics'] 
                 for a in all_factors.loc[:, 'category']], 'factor'].tolist()

# ============================================================================
# 辅助函数
# ============================================================================

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
    vwap_ret_series = vwap_prices.iloc[-1] / vwap_prices.iloc[0] - 1
    return vwap_ret_series


def cal_portfolio_vwap_ret(old_portfolio_weight_series, new_portfolio_weight_series, buy_date, sell_date):
    """
    计算一个组合从买入到卖出的持有期回报率
    如果已经有旧头寸组合，则从旧头寸调仓到新头寸的成本也计算在本次持有期回报内
    """
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
    """标准化因子序列"""
    series = series.copy().replace([np.inf, -np.inf], np.nan)
    
    # 对于极端值使用对数变换
    if series.abs().max() > 101:
        series = np.sign(series) * np.log2(1. + series.abs())

    # 处理无效数据
    if np.isnan(series.mean()) or np.isnan(series.std()) or (series.std() < 0.000001): 
        series.iloc[:] = 0.
        return series

    # 对于低基数因子直接标准化
    if len(series.unique()) <= 20: 
        series = (series - series.mean()) / series.std()
        return series.fillna(0)

    # Winsorization：对极端值进行截尾处理
    q = series.quantile([0.01, 0.99])
    series[series < q.iloc[0]] = q.iloc[0]
    series[series > q.iloc[1]] = q.iloc[1]

    # 标准化
    series = (series - series.mean()) / series.std()
    series = series.replace([np.inf, -np.inf], np.nan).fillna(0.)
    
    return series


def get_my_factors(decision_date, all_stocks):
    """获取指定日期和股票的所有因子"""
    factor_df_list = []
    for i_factor in all_factors:
        factor_df_list.append(get_factor_values(securities=all_stocks, 
                              factors=i_factor, 
                              start_date=decision_date, 
                              end_date=decision_date)[i_factor].T)
    factor_df = pd.concat(factor_df_list, axis=1)
    factor_df.columns = all_factors
    return factor_df


# ============================================================================
# 模型训练 - 改进版
# ============================================================================

def train_improved_model():
    """
    训练改进的模型以提高信息比率
    使用start_date之前的数据进行训练
    """
    print("开始准备训练数据...")
    
    # 训练数据使用start_date之前3年的数据
    training_dates = get_buy_dates(
        start_date=start_date - datetime.timedelta(365*3), 
        end_date=start_date, 
        freq=investment_horizon
    )
    
    print(f"训练数据时间范围: {training_dates[0]} 到 {training_dates[-1]}")
    print(f"训练数据周期数: {len(training_dates)-1}")
    
    # 准备训练数据
    factor_df_list = []
    for i in tqdm(range(len(training_dates)-1), desc="准备训练数据"):
        buy_date = training_dates[i]
        sell_date = training_dates[i+1]
        i_pre_date = get_previous_trade_date(buy_date)
        
        all_stocks = get_index_stocks('000852.XSHG', date=i_pre_date)
        all_stocks = list(set(all_stocks) - get_st_or_paused_stock_set(i_pre_date))
        
        factor_df = get_my_factors(i_pre_date, all_stocks)
        factor_df.loc[:, 'next_vwap_ret'] = cal_vwap_ret_series(all_stocks, buy_date, sell_date)
        factor_df = factor_df.apply(normalize_series)
        factor_df_list.append(factor_df)
    
    factor_df_all = pd.concat(factor_df_list).dropna()
    print(f"训练样本数: {len(factor_df_all)}")
    
    X_train = factor_df_all.iloc[:, :-1]
    y_train = factor_df_all.iloc[:, -1]
    
    # ========================================================================
    # 模型1: 增强的Ridge回归 (带特征工程)
    # ========================================================================
    print("\n训练模型1: Ridge回归 (基准模型)...")
    from sklearn.linear_model import Ridge
    model_ridge = Ridge(alpha=1.0)
    model_ridge.fit(X_train, y_train)
    
    # ========================================================================
    # 模型2: Lasso回归 (特征选择)
    # ========================================================================
    print("训练模型2: Lasso回归 (特征选择)...")
    from sklearn.linear_model import Lasso
    model_lasso = Lasso(alpha=0.001, max_iter=10000)
    model_lasso.fit(X_train, y_train)
    
    # ========================================================================
    # 模型3: ElasticNet (结合L1和L2正则化)
    # ========================================================================
    print("训练模型3: ElasticNet...")
    from sklearn.linear_model import ElasticNet
    model_elastic = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)
    model_elastic.fit(X_train, y_train)
    
    # ========================================================================
    # 模型4: Random Forest (非线性模型)
    # ========================================================================
    print("训练模型4: Random Forest...")
    from sklearn.ensemble import RandomForestRegressor
    model_rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    model_rf.fit(X_train, y_train)
    
    # ========================================================================
    # 模型5: Gradient Boosting (强大的集成模型)
    # ========================================================================
    print("训练模型5: Gradient Boosting...")
    from sklearn.ensemble import GradientBoostingRegressor
    model_gb = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42
    )
    model_gb.fit(X_train, y_train)
    
    # ========================================================================
    # 创建集成模型 (Ensemble)
    # ========================================================================
    print("\n创建集成模型...")
    
    class EnsembleModel:
        """集成多个模型的预测结果"""
        def __init__(self, models, weights=None):
            self.models = models
            if weights is None:
                self.weights = [1.0 / len(models)] * len(models)
            else:
                self.weights = weights
        
        def predict(self, X):
            predictions = np.zeros(len(X))
            for model, weight in zip(self.models, self.weights):
                predictions += weight * model.predict(X)
            return predictions
        
        def fit(self, X, y):
            # 集成模型已经包含了训练好的子模型
            pass
    
    # 集成所有模型，权重可以根据验证集表现调整
    # 这里使用等权重，实际应用中可以根据验证集IC优化权重
    ensemble_model = EnsembleModel(
        models=[model_ridge, model_lasso, model_elastic, model_rf, model_gb],
        weights=[0.15, 0.15, 0.15, 0.25, 0.30]  # GB和RF权重稍高
    )
    
    print("\n模型训练完成!")
    print("=" * 60)
    print("使用的模型组合:")
    print("  - Ridge回归 (15%)")
    print("  - Lasso回归 (15%)")
    print("  - ElasticNet (15%)")
    print("  - Random Forest (25%)")
    print("  - Gradient Boosting (30%)")
    print("=" * 60)
    
    return ensemble_model


# 训练模型
print("=" * 60)
print("开始训练my_model...")
print(f"投资周期 (investment_horizon): {investment_horizon}")
print(f"每年周期数: {number_of_periods_per_year}")
print(f"模拟文件: {simulation_file}")
print("=" * 60)

my_model = train_improved_model()


# ============================================================================
# 策略核心函数 (不能修改)
# ============================================================================

def cal_portfolio_weight_series(decision_date, old_portfolio_weight_series):
    """
    在decision_date收盘后决定接下来下一天要买的投资组合
    
    注意：
    - 这个函数里面不能使用任何decision_date所在时间之后的信息
    - decision_date当天收盘的信息依然可用，假设在decision_date的下一天才调仓
    """
    # 基础股票池，去掉st和停牌股
    all_stocks = get_index_stocks('000852.XSHG', date=decision_date)
    all_stocks = list(set(all_stocks) - get_st_or_paused_stock_set(decision_date))
    
    # 抓取因子
    factor_df = get_my_factors(decision_date, all_stocks)
    factor_df = factor_df.apply(normalize_series)
    
    # 使用模型预测
    predicted_factor = pd.Series(my_model.predict(factor_df), index=factor_df.index)
    
    # 筛选：选择预测得分最高的50只股票
    filtered_assets = predicted_factor.nlargest(50).index.tolist()
    
    # 等权配权
    portfolio_weight_series = pd.Series(1/len(filtered_assets), index=filtered_assets)
    
    return portfolio_weight_series


# ============================================================================
# 回测模拟
# ============================================================================

def simulate_wealth_process(start_date, end_date):
    """模拟一段时间的策略回测"""
    print("\n开始回测模拟...")
    
    all_buy_dates = get_buy_dates(start_date, end_date, investment_horizon)
    wealth_process = pd.Series(np.nan, index=all_buy_dates)
    wealth_process.iloc[0] = 1
    allocation_dict = dict()
    old_portfolio_weight_series = pd.Series(dtype='float64')
    
    # 断点续跑机制
    start_index = 0
    if os.path.exists(simulation_file):
        print(f"发现断点文件 {simulation_file}，继续之前的模拟...")
        start_index, allocation_dict, wealth_process, old_portfolio_weight_series = \
            pickle.load(open(simulation_file, 'rb'))
    
    for i in tqdm(range(start_index, len(all_buy_dates)-1), desc="回测进度"):
        buy_date = all_buy_dates[i]
        sell_date = all_buy_dates[i+1]
        
        decision_date = get_previous_trade_date(buy_date)
        new_portfolio_weight_series = cal_portfolio_weight_series(decision_date, old_portfolio_weight_series)
        
        allocation_dict[buy_date] = new_portfolio_weight_series.copy()
        
        wealth_process.loc[sell_date] = wealth_process.loc[buy_date] * \
            (1 + cal_portfolio_vwap_ret(old_portfolio_weight_series, new_portfolio_weight_series, buy_date, sell_date))
        
        old_portfolio_weight_series = new_portfolio_weight_series
        
        # 保存数据进行断点续跑
        pickle.dump([i+1, allocation_dict, wealth_process, old_portfolio_weight_series], 
                    open(simulation_file, "wb"), protocol=4)
    
    print("回测模拟完成!")
    return wealth_process, allocation_dict


# 进行策略回测模拟
wealth_process, allocation_dict = simulate_wealth_process(start_date, end_date)


# ============================================================================
# 性能评估
# ============================================================================

def evaluate_performance():
    """评估策略性能"""
    print("\n" + "=" * 60)
    print("性能评估")
    print("=" * 60)
    
    # 获取基准ETF数据
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
    wealth_process.name = '我的策略'
    combined_df = pd.concat([wealth_process, benchmark_index], axis=1)
    
    # 计算收益率
    combined_df_ret = combined_df.pct_change()
    combined_df_ret.loc[:, '超额收益'] = combined_df_ret.loc[:, '我的策略'] - combined_df_ret.loc[:, '基准ETF']
    
    # 计算夏普比率和信息比率
    performance_metrics = combined_df_ret.mean() * number_of_periods_per_year / \
        (combined_df_ret.std() * sqrt(number_of_periods_per_year))
    performance_metrics.index = ['我的策略夏普', '基准ETF夏普', '信息比率']
    
    print("\n性能指标:")
    print(performance_metrics)
    print("\n" + "=" * 60)
    
    return combined_df, performance_metrics


# 评估性能
combined_df, performance_metrics = evaluate_performance()

print("\n策略执行完成!")
print(f"信息比率: {performance_metrics['信息比率']:.4f}")
