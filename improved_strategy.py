import datetime
import os

import numpy as np
import pandas as pd
from jqlib import alpha101  # noqa: F401  # kept for parity with original notebook
from jqfactor import get_all_factors, get_factor_values
from sklearn.linear_model import Ridge
from tqdm import tqdm
import jqdata
from jqdata import (
    get_all_securities,
    get_all_trade_days,
    get_extras,
    get_index_stocks,
    get_price,
    get_trade_days,
)

# -----------------------------
# 参数配置
# -----------------------------
start_date = datetime.date(2020, 1, 1)
end_date = datetime.date(2025, 12, 15)
investment_horizon = "M"  # 只允许 'M' 或 'W'

if investment_horizon not in ("M", "W"):
    raise ValueError("investment_horizon must be 'M' or 'W'")

number_of_periods_per_year = 12 if investment_horizon == "M" else 52
simulation_file = "L10_temp_fixed_m_basicsrisk.pkl"

# 重新设计模型时清理断点续跑文件，防止复用旧状态
if os.path.exists(simulation_file):
    os.remove(simulation_file)


# -----------------------------
# 辅助函数
# -----------------------------
def get_st_or_paused_stock_set(decision_date):
    all_stock_ids = get_all_securities(types=["stock"], date=decision_date).index.tolist()
    is_st_flag = get_extras("is_st", all_stock_ids, start_date=decision_date, end_date=decision_date)
    st_set = set(is_st_flag.iloc[0][is_st_flag.iloc[0]].index)

    paused_flag = get_price(
        all_stock_ids,
        start_date=decision_date,
        end_date=decision_date,
        frequency="daily",
        fq="post",
        panel=False,
        fields=["paused"],
    )
    paused_set = set(paused_flag.loc[paused_flag.loc[:, "paused"] == 1].loc[:, "code"].values)
    return st_set.union(paused_set)


def cal_vwap_ret_series(order_book_ids, buy_date, sell_date):
    if len(order_book_ids) == 0:
        return pd.Series(dtype="float64")
    all_data = get_price(
        order_book_ids,
        buy_date,
        sell_date,
        fields=["money", "volume"],
        fq="post",
        panel=False,
    )
    vwap_prices = all_data.set_index(["time", "code"]).unstack().loc[:, "money"] / all_data.set_index(
        ["time", "code"]
    ).unstack().loc[:, "volume"]
    vwap_ret_series = vwap_prices.iloc[-1] / vwap_prices.iloc[0] - 1
    return vwap_ret_series


def cal_portfolio_vwap_ret(old_portfolio_weight_series, new_portfolio_weight_series, buy_date, sell_date):
    order_book_ids = new_portfolio_weight_series.index.tolist()

    vwap_ret_series = cal_vwap_ret_series(order_book_ids, buy_date, sell_date)
    hpr = (new_portfolio_weight_series * vwap_ret_series).sum()

    old_new = pd.concat([old_portfolio_weight_series, new_portfolio_weight_series], axis=1).fillna(0)
    cost = (old_new.iloc[:, 0] - old_new.iloc[:, 1]).abs().sum() * 0.001

    return hpr - cost


def get_previous_trade_date(current_date):
    trading_dates = get_all_trade_days()
    trading_dates = sorted(trading_dates)
    for trading_date in reversed(trading_dates):
        if trading_date < current_date:
            return trading_date
    return None


def get_next_trade_date(current_date):
    trading_dates = get_all_trade_days()
    trading_dates = sorted(trading_dates)
    for trading_date in trading_dates:
        if trading_date > current_date:
            return trading_date
    return None


def get_buy_dates(start_date: str, end_date: str, freq: str) -> list:
    periodic_dates = [x.date() for x in pd.date_range(start_date, end_date, freq=freq)]
    trading_dates = get_trade_days(start_date, end_date)
    return np.sort(
        np.unique([get_next_trade_date(d) for d in periodic_dates if (get_next_trade_date(d) <= end_date)])
    ).tolist()


def normalize_series(series):
    series = series.copy().replace([np.inf, -np.inf], np.nan)

    if series.abs().max() > 101:
        series = np.sign(series) * np.log2(1.0 + series.abs())

    if np.isnan(series.mean()) or np.isnan(series.std()) or (series.std() < 0.000001):
        series.iloc[:] = 0.0
        return series

    if len(series.unique()) <= 20:
        series = (series - series.mean()) / series.std()
        return series.fillna(0)

    q = series.quantile([0.01, 0.99])
    series[series < q.iloc[0]] = q.iloc[0]
    series[series > q.iloc[1]] = q.iloc[1]

    series = (series - series.mean()) / series.std()

    series = series.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return series


all_factors = get_all_factors()
all_factors = all_factors.loc[
    [a in ["risk", "basics"] for a in all_factors.loc[:, "category"]], "factor"
].tolist()


def get_my_factors(decision_date, all_stocks):
    factor_df_list = []
    for i_factor in all_factors:
        factor_df_list.append(
            get_factor_values(
                securities=all_stocks,
                factors=i_factor,
                start_date=decision_date,
                end_date=decision_date,
            )[i_factor].T
        )
    factor_df = pd.concat(factor_df_list, axis=1)
    factor_df.columns = all_factors
    return factor_df


# -----------------------------
# 训练模型（使用 start_date 之前的数据）
# -----------------------------
training_cutoff_date = get_previous_trade_date(start_date)
if training_cutoff_date is None:
    raise ValueError("无法确定 start_date 之前的交易日，用于训练集划分。")

training_dates = get_buy_dates(
    start_date=start_date - datetime.timedelta(365 * 3),
    end_date=training_cutoff_date,
    freq=investment_horizon,
)

factor_df_list = []
for i in tqdm(range(len(training_dates) - 1)):
    buy_date = training_dates[i]
    sell_date = training_dates[i + 1]
    i_pre_date = get_previous_trade_date(buy_date)
    all_stocks = get_index_stocks("000852.XSHG", date=i_pre_date)
    all_stocks = list(set(all_stocks) - get_st_or_paused_stock_set(i_pre_date))

    factor_df = get_my_factors(i_pre_date, all_stocks)
    factor_df.loc[:, "next_vwap_ret"] = cal_vwap_ret_series(all_stocks, buy_date, sell_date)
    factor_df = factor_df.apply(normalize_series)
    factor_df_list.append(factor_df)

factor_df_list = pd.concat(factor_df_list).dropna()

my_model = Ridge()
my_model.fit(factor_df_list.iloc[:, :-1], factor_df_list.iloc[:, -1])


# -----------------------------
# 投组构建逻辑（保持与 L10 一致）
# -----------------------------
def cal_portfolio_weight_series(decision_date, old_portfolio_weight_series):
    # 在decision_date收盘后决定接下来下一天要买的投资组合
    # 这个函数里面不能使用任何decision_date所在时间之后的信息
    #（decision_date当天收盘的信息依然可用，假设在decision_date的下一天才调仓）

    # 基础股票池，去掉st和停牌股
    all_stocks = get_index_stocks("000852.XSHG", date=decision_date)
    all_stocks = list(set(all_stocks) - get_st_or_paused_stock_set(decision_date))

    # 抓取因子
    factor_df = get_my_factors(decision_date, all_stocks)
    factor_df = factor_df.apply(normalize_series)

    # 预测
    predicted_factor = pd.Series(my_model.predict(factor_df), index=factor_df.index)

    # 筛选
    filtered_assets = predicted_factor.nlargest(50).index.tolist()

    # 配权
    portfolio_weight_series = pd.Series(1 / len(filtered_assets), index=filtered_assets)

    return portfolio_weight_series


# -----------------------------
# 回测逻辑（不再使用断点续跑文件）
# -----------------------------
def simulate_wealth_process(start_date, end_date):
    all_buy_dates = get_buy_dates(start_date, end_date, investment_horizon)
    wealth_process = pd.Series(np.nan, index=all_buy_dates)
    wealth_process.iloc[0] = 1
    allocation_dict = dict()
    old_portfolio_weight_series = pd.Series(dtype="float64")

    for i in tqdm(range(len(all_buy_dates) - 1)):  # 这里循环只到倒数第二个买卖日
        buy_date = all_buy_dates[i]
        sell_date = all_buy_dates[i + 1]

        decision_date = get_previous_trade_date(buy_date)  # 前一天晚上做决策，在buy_date用vwap价格买卖
        new_portfolio_weight_series = cal_portfolio_weight_series(decision_date, old_portfolio_weight_series)

        allocation_dict[buy_date] = new_portfolio_weight_series.copy()  # 这里的copy巨重要

        wealth_process.loc[sell_date] = wealth_process.loc[buy_date] * (
            1 + cal_portfolio_vwap_ret(old_portfolio_weight_series, new_portfolio_weight_series, buy_date, sell_date)
        )

        old_portfolio_weight_series = new_portfolio_weight_series

    return wealth_process, allocation_dict


if __name__ == "__main__":
    wealth_process, allocation_dict = simulate_wealth_process(start_date, end_date)
    print(wealth_process.dropna().tail())
