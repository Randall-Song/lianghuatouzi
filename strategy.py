"""
多因子量化投资策略 - Multi-Factor Quantitative Investment Strategy

本策略使用Gradient Boosting模型预测股票收益，基于49个风险和基本面因子。
目标是最大化投资组合的信息比率（Information Ratio）。

主要特点：
1. 使用GradientBoostingRegressor捕捉非线性关系和因子交互
2. 训练数据使用start_date之前的历史数据
3. 支持月度(M)和周度(W)调仓
4. 包含模型持久化和回测断点续跑机制

适用环境：聚宽(JoinQuant)量化平台
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
from sklearn.ensemble import GradientBoostingRegressor
from math import sqrt

# ============================================================================
# Configuration
# ============================================================================
start_date = datetime.date(2020, 1, 1)
end_date = datetime.date(2025, 12, 15)
investment_horizon = 'M'  # M 为月度调仓， W为周度调仓, d为日度调仓
number_of_periods_per_year = 12  # 一年12个交易月，52个交易周，252个交易日
simulation_file = "L10_temp_fixed_m_basicsrisk.pkl"
model_file = "my_model.pkl"

# Model configuration
TRAINING_YEARS = 4  # 训练数据的历史年数
N_STOCKS = 40  # 投资组合中的股票数量

# GradientBoosting hyperparameters
GB_N_ESTIMATORS = 150
GB_LEARNING_RATE = 0.03
GB_MAX_DEPTH = 5
GB_MIN_SAMPLES_SPLIT = 30
GB_MIN_SAMPLES_LEAF = 15
GB_SUBSAMPLE = 0.7
GB_MAX_FEATURES = 'sqrt'
GB_LOSS = 'huber'
GB_ALPHA = 0.9
GB_RANDOM_STATE = 42

# ============================================================================
# Get all risk and basics factors
# ============================================================================
all_factors = get_all_factors()
all_factors = all_factors.loc[[a in ['risk', 'basics'] \
                 for a in all_factors.loc[:, 'category']], 'factor'].tolist()

# ============================================================================
# Utility Functions
# ============================================================================

def get_st_or_paused_stock_set(decision_date):
    # 抓取st和停牌股票的集合
    all_stock_ids = get_all_securities(types=['stock'], date=decision_date).index.tolist()
    is_st_flag = get_extras('is_st', all_stock_ids, start_date=decision_date, end_date=decision_date)
    st_set = set(is_st_flag.iloc[0][is_st_flag.iloc[0]].index)
    
    paused_flag = get_price(all_stock_ids, 
                        start_date = decision_date, 
                        end_date = decision_date, 
                        frequency='daily',
                        fq='post', panel=False, fields=['paused'])
    paused_set = set(paused_flag.loc[paused_flag.loc[:, 'paused']==1].loc[:, 'code'].values)
    return st_set.union(paused_set)

def cal_vwap_ret_series(order_book_ids, buy_date, sell_date):
    # 计算一组资产买卖周期内的vwap回报率。
    if len(order_book_ids)==0: return pd.Series(dtype='float64')
    all_data = get_price(order_book_ids, buy_date, sell_date, fields=['money', 'volume'], 
                            fq='post', panel=False)
    vwap_prices = all_data.set_index(['time', 'code']).unstack().loc[:, 'money'] / \
        all_data.set_index(['time', 'code']).unstack().loc[:, 'volume']
    vwap_ret_series = vwap_prices.iloc[-1]/vwap_prices.iloc[0]-1
    return vwap_ret_series

def cal_portfolio_vwap_ret(old_portfolio_weight_series, new_portfolio_weight_series, buy_date, sell_date):
    # 计算一个组合从买入到卖出的持有期回报率
    # 如果已经有旧头寸组合，则从旧头寸调仓到新头寸的成本也计算在本次持有期回报内。
    order_book_ids = new_portfolio_weight_series.index.tolist()
    
    vwap_ret_series = cal_vwap_ret_series(order_book_ids, buy_date, sell_date)
    hpr = (new_portfolio_weight_series * vwap_ret_series).sum()
    
    old_new = pd.concat([old_portfolio_weight_series, new_portfolio_weight_series], axis=1).fillna(0)
    cost = (old_new.iloc[:, 0] - old_new.iloc[:, 1]).abs().sum()*0.001
    
    return hpr-cost

def get_previous_trade_date(current_date):
    # 抓取上一交易日
    trading_dates = jqdata.get_all_trade_days()
    trading_dates = sorted(trading_dates)
    for trading_date in reversed(trading_dates):
        if trading_date < current_date:
            return trading_date
    return None

def get_next_trade_date(current_date):
    # 抓取下一交易日
    trading_dates = jqdata.get_all_trade_days()
    trading_dates = sorted(trading_dates)
    for trading_date in trading_dates:
        if trading_date > current_date:
            return trading_date
    return None

def get_buy_dates(start_date: str, end_date: str, freq: str) -> list:
    # 抓取要进行买卖调仓的日期
    periodic_dates = [x.date() for x in pd.date_range(start_date, end_date, freq=freq)]
    trading_dates = jqdata.get_trade_days(start_date, end_date)
    return np.sort(np.unique([get_next_trade_date(d) for d in periodic_dates \
                              if (get_next_trade_date(d) <= end_date)])).tolist()

def normalize_series(series):
    series = series.copy().replace([np.inf, -np.inf], np.nan)
    
    if series.abs().max()>101:
        series = np.sign(series) * np.log2(1. + series.abs())

    if np.isnan(series.mean()) or np.isnan(series.std()) or (series.std()<0.000001): 
        series.iloc[:] = 0.
        return series

    if len(series.unique()) <= 20: 
        series = (series - series.mean()) / series.std()
        return series.fillna(0)

    q = series.quantile([0.01, 0.99])
    series[series<q.iloc[0]] = q.iloc[0]
    series[series>q.iloc[1]] = q.iloc[1]

    series = (series - series.mean()) / series.std()
    
    series = series.replace([np.inf, -np.inf], np.nan).fillna(0.)
    
    return series

def get_my_factors(decision_date, all_stocks):
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
# Model Training
# ============================================================================

def train_my_model():
    """
    Train an improved model with better feature engineering and model selection
    Training data must be from before start_date
    """
    print("开始训练模型...")
    
    # 训练数据使用的时间节点 - 使用start_date之前的数据
    training_dates = get_buy_dates(start_date = start_date - datetime.timedelta(days=365*TRAINING_YEARS), 
                                   end_date = start_date, freq=investment_horizon)
    
    print(f"训练期数: {len(training_dates)-1}")
    
    factor_df_list = []
    for i in tqdm(range(len(training_dates)-1)):
        buy_date = training_dates[i]
        sell_date = training_dates[i+1]
        i_pre_date = get_previous_trade_date(buy_date)
        all_stocks = get_index_stocks('000852.XSHG', date=i_pre_date)
        all_stocks = list(set(all_stocks) - get_st_or_paused_stock_set(i_pre_date))
        
        factor_df = get_my_factors(i_pre_date, all_stocks)
        
        factor_df.loc[:, 'next_vwap_ret'] = cal_vwap_ret_series(all_stocks, buy_date, sell_date)
        factor_df = factor_df.apply(normalize_series)
        factor_df_list.append(factor_df)
    
    factor_df_list = pd.concat(factor_df_list).dropna()
    
    print(f"训练样本数: {len(factor_df_list)}")
    
    # 使用改进的模型 - Gradient Boosting with optimized hyperparameters
    # This model can capture non-linear relationships and interactions better than Ridge
    my_model = GradientBoostingRegressor(
        n_estimators=GB_N_ESTIMATORS,
        learning_rate=GB_LEARNING_RATE,
        max_depth=GB_MAX_DEPTH,
        min_samples_split=GB_MIN_SAMPLES_SPLIT,
        min_samples_leaf=GB_MIN_SAMPLES_LEAF,
        subsample=GB_SUBSAMPLE,
        max_features=GB_MAX_FEATURES,
        random_state=GB_RANDOM_STATE,
        loss=GB_LOSS,
        alpha=GB_ALPHA
    )
    
    X_train = factor_df_list.iloc[:, :-1]
    y_train = factor_df_list.iloc[:, -1]
    
    my_model.fit(X_train, y_train)
    
    # 保存模型
    with open(model_file, 'wb') as f:
        pickle.dump(my_model, f, protocol=4)
    
    print("模型训练完成并保存!")
    
    # 输出特征重要性
    feature_importance = pd.Series(my_model.feature_importances_, index=all_factors)
    print("\n前10个最重要的因子:")
    print(feature_importance.nlargest(10))
    
    return my_model

def load_or_train_model():
    """
    Load existing model or train a new one
    """
    if os.path.exists(model_file):
        print(f"加载已保存的模型: {model_file}")
        with open(model_file, 'rb') as f:
            my_model = pickle.load(f)
        return my_model
    else:
        print("未找到已保存的模型，开始训练新模型...")
        return train_my_model()

# Load or train the model
my_model = load_or_train_model()

# ============================================================================
# Portfolio Construction
# ============================================================================

def cal_portfolio_weight_series(decision_date, old_portfolio_weight_series):
    """
    在decision_date收盘后决定接下来下一天要买的投资组合
    这个函数里面不能使用任何decision_date所在时间之后的信息
    （decision_date当天收盘的信息依然可用，假设在decision_date的下一天才调仓）
    
    注意: 此函数不能改动其接口
    """
    
    # 基础股票池，去掉st和停牌股
    all_stocks = get_index_stocks('000852.XSHG', date=decision_date)
    all_stocks = list(set(all_stocks) - get_st_or_paused_stock_set(decision_date))
    
    # 抓取因子
    factor_df = get_my_factors(decision_date, all_stocks)
    factor_df = factor_df.apply(normalize_series)
    
    # 预测
    predicted_factor = pd.Series(my_model.predict(factor_df), index = factor_df.index)
    
    # 筛选 - 选择预测收益最高的股票
    filtered_assets = predicted_factor.nlargest(N_STOCKS).index.tolist()
    
    # 配权 - 使用预测值加权而不是等权，给予更高预测收益的股票更高权重
    predicted_weights = predicted_factor.loc[filtered_assets]
    # 将预测值转换为权重（使用softmax-like transformation）
    predicted_weights = predicted_weights - predicted_weights.min() + 0.01  # Ensure all positive
    predicted_weights = predicted_weights / predicted_weights.sum()  # Normalize to sum to 1
    
    portfolio_weight_series = predicted_weights
    
    return portfolio_weight_series

# ============================================================================
# Backtesting
# ============================================================================

def simulate_wealth_process(start_date, end_date):
    """
    模拟一段时间的策略回测
    支持断点续跑机制
    """
    
    all_buy_dates = get_buy_dates(start_date, end_date, investment_horizon)
    wealth_process = pd.Series(np.nan, index=all_buy_dates)
    wealth_process.iloc[0] = 1
    allocation_dict = dict()
    old_portfolio_weight_series = pd.Series(dtype='float64')
    
    # 断点续跑机制
    start_index = 0
    if os.path.exists(simulation_file):
        start_index, allocation_dict, wealth_process, old_portfolio_weight_series = \
            pickle.load(open(simulation_file, 'rb'))
        print(f"从断点 {start_index}/{len(all_buy_dates)-1} 继续运行")
            
    for i in tqdm(range(start_index, len(all_buy_dates)-1)): # 这里循环只到倒数第二个买卖日
        buy_date = all_buy_dates[i]
        sell_date = all_buy_dates[i+1]
        
        decision_date = get_previous_trade_date(buy_date) # 前一天晚上做决策，在buy_date用vwap价格买卖
        new_portfolio_weight_series = cal_portfolio_weight_series(decision_date, old_portfolio_weight_series)
        
        allocation_dict[buy_date] = new_portfolio_weight_series.copy() # 这里的copy巨重要
        
        wealth_process.loc[sell_date] = wealth_process.loc[buy_date] * \
            (1+cal_portfolio_vwap_ret(old_portfolio_weight_series, new_portfolio_weight_series, buy_date, sell_date))
        
        old_portfolio_weight_series = new_portfolio_weight_series
        
        # 保存数据进行断点续跑
        pickle.dump([i+1, allocation_dict, wealth_process, old_portfolio_weight_series], \
                    open(simulation_file, "wb"), protocol=4)
        
    return wealth_process, allocation_dict

# ============================================================================
# Performance Evaluation
# ============================================================================

def evaluate_performance(wealth_process):
    """
    计算策略的表现指标
    """
    # 获取基准ETF的日度vwap价格
    benchmark_index = get_price(['512100.XSHG'], 
                start_date = start_date, 
                end_date = end_date, 
                frequency='daily',
                fq='post', panel=False, fields=['money', 'volume'])
    benchmark_index.loc[:, 'vwap'] = benchmark_index.loc[:, 'money'] / benchmark_index.loc[:, 'volume']
    
    benchmark_index = benchmark_index.set_index('time').loc[:, ['vwap']].loc[wealth_process.index]
    benchmark_index = benchmark_index/benchmark_index.iloc[0]
    benchmark_index.columns = ['基准ETF']
    
    wealth_process.name = '我的策略'
    combined_df = pd.concat([wealth_process, benchmark_index], axis=1)
    
    # 计算收益率
    combined_df_ret = combined_df.pct_change()
    combined_df_ret.loc[:, '超额收益'] = combined_df_ret.loc[:, '我的策略'] - combined_df_ret.loc[:, '基准ETF']
    
    # 计算性能指标
    performance_metrics = combined_df_ret.mean() * number_of_periods_per_year / \
        (combined_df_ret.std()*sqrt(number_of_periods_per_year))
    performance_metrics.index = ['我的策略夏普', '基准ETF夏普', '信息比率']
    
    return performance_metrics, combined_df

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("量化投资策略 - 多因子模型")
    print("="*80)
    
    # 进行策略回测模拟
    print("\n开始回测...")
    wealth_process, allocation_dict = simulate_wealth_process(start_date, end_date)
    
    # 评估性能
    print("\n评估策略表现...")
    performance_metrics, combined_df = evaluate_performance(wealth_process)
    
    print("\n" + "="*80)
    print("策略表现:")
    print("="*80)
    print(performance_metrics)
    print("\n信息比率: {:.6f}".format(performance_metrics['信息比率']))
    print("="*80)
