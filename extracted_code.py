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

start_date = datetime.date(2020, 1, 1)
end_date = datetime.date(2025, 12, 15)
investment_horizon = 'M' # M 为月度调仓， W为周度调仓, d为日度调仓
number_of_periods_per_year = 12 # 一年12个交易月，52个交易周，252个交易日
simulation_file = "L10_temp_fixed_m_basicsrisk.pkl"

all_factors = get_all_factors()
all_factors = all_factors.loc[[a in ['risk', 'basics'] \
                 for a in all_factors.loc[:, 'category']], 'factor'].tolist()
len(all_factors)

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

# 交易日相关辅助函数

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

# 训练数据使用的时间节点
training_dates = get_buy_dates(start_date = start_date - datetime.timedelta(365*3), \
                               end_date = start_date, freq=investment_horizon)

buy_date = training_dates[0]
sell_date = training_dates[1]
i_pre_date = get_previous_trade_date(buy_date)
all_stocks = get_index_stocks('000852.XSHG', date=i_pre_date)
all_stocks = list(set(all_stocks) - get_st_or_paused_stock_set(i_pre_date))
factor_df = get_my_factors(i_pre_date, all_stocks)
factor_df.loc[:, 'next_vwap_ret'] = cal_vwap_ret_series(all_stocks, buy_date, sell_date)
factor_df = factor_df.apply(normalize_series)
factor_df.head()

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

factor_df_list.head()

from sklearn.linear_model import Ridge

my_model = Ridge()
my_model.fit(factor_df_list.iloc[:, :-1], factor_df_list.iloc[:, -1])
coefficients = my_model.coef_

print("回归权重（系数）:")
for i, coef in enumerate(coefficients):
    print(f"X{i}的系数: {coef}")

pd.Series(my_model.predict(factor_df_list.iloc[:, :-1]), index = factor_df_list.index).head()

def cal_portfolio_weight_series(decision_date, old_portfolio_weight_series):
    # 在decision_date收盘后决定接下来下一天要买的投资组合
    # 这个函数里面不能使用任何decision_date所在时间之后的信息
    #（decision_date当天收盘的信息依然可用，假设在decision_date的下一天才调仓）
    
    # 基础股票池，去掉st和停牌股
    all_stocks = get_index_stocks('000852.XSHG', date=decision_date)
    all_stocks = list(set(all_stocks) - get_st_or_paused_stock_set(decision_date))
    
    # 抓取因子
    factor_df = get_my_factors(decision_date, all_stocks)
    factor_df = factor_df.apply(normalize_series)
    
    # 预测
    predicted_factor = pd.Series(my_model.predict(factor_df), index = factor_df.index)
    
    # 筛选
    filtered_assets = predicted_factor.nlargest(50).index.tolist()
    
    # 配权
    portfolio_weight_series = pd.Series(1/len(filtered_assets), index=filtered_assets)
    
    return portfolio_weight_series

def simulate_wealth_process(start_date, end_date):
    # 模拟一段时间的策略回测
    
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
            
    for i in tqdm(range(start_index, len(all_buy_dates)-1)): # 这里循环只到倒数第二个买卖日
        buy_date = all_buy_dates[i]
        sell_date = all_buy_dates[i+1]
        
        decision_date = get_previous_trade_date(buy_date) # 前一天晚上做决策，在buy_date用vwap价格买卖
        new_portfolio_weight_series = cal_portfolio_weight_series(decision_date, old_portfolio_weight_series) # 这里比之前增加了参数
        
        allocation_dict[buy_date] = new_portfolio_weight_series.copy() # 这里的copy巨重要
        
        wealth_process.loc[sell_date] = wealth_process.loc[buy_date] * \
            (1+cal_portfolio_vwap_ret(old_portfolio_weight_series, new_portfolio_weight_series, buy_date, sell_date))
        
        old_portfolio_weight_series = new_portfolio_weight_series
        
        # 保存数据进行断点续跑
        pickle.dump([i+1, allocation_dict, wealth_process, old_portfolio_weight_series], \
                    open(simulation_file, "wb"), protocol=4)
        
    return wealth_process, allocation_dict

# 进行策略回测模拟
wealth_process, allocation_dict = simulate_wealth_process(start_date, end_date)

# 查看不同时间的选股个数
pd.Series([len(b) for a,b in allocation_dict.items()]).plot(figsize=(10, 3))

# 画策略的财富曲线
wealth_process.name = '我的策略'
wealth_process.plot(figsize=(10, 3))

get_security_info('512100.XSHG').display_name

# 获取基准ETF的日度vwap价格，注意不能直接用指数 000852.XSHG !!
benchmark_index = get_price(['512100.XSHG'], 
            start_date = start_date, 
            end_date = end_date, 
            frequency='daily',
            fq='post', panel=False, fields=['money', 'volume'])
benchmark_index.loc[:, 'vwap'] = benchmark_index.loc[:, 'money'] / benchmark_index.loc[:, 'volume']
benchmark_index.head()

benchmark_index = benchmark_index.set_index('time').loc[:, ['vwap']].loc[wealth_process.index]
benchmark_index.head()

benchmark_index = benchmark_index/benchmark_index.iloc[0]
benchmark_index.columns = ['基准ETF']
benchmark_index.head()

combined_df = pd.concat([wealth_process, benchmark_index], axis=1)
combined_df.plot(figsize=(10, 3))

combined_df_ret = combined_df.pct_change()
combined_df_ret.loc[:, '超额收益'] = combined_df_ret.loc[:, '我的策略'] - combined_df_ret.loc[:, '基准ETF']
performance_metrics = combined_df_ret.mean() * number_of_periods_per_year / \
    (combined_df_ret.std()*sqrt(number_of_periods_per_year))
performance_metrics.index = ['我的策略夏普', '基准ETF夏普', '信息比率']
performance_metrics

fig = plt.figure(figsize=(3, 3))
axes = fig.add_axes([0, 0, 1, 1])
axes.bar(performance_metrics.index, performance_metrics)

(1+combined_df_ret.fillna(0)).cumprod().plot(figsize=(10, 3))