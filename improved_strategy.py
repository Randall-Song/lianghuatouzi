import datetime
import os

import numpy as np
import pandas as pd
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
)

# -----------------------------
# 参数配置
# -----------------------------
start_date = datetime.date(2020, 1, 1)
END_DATE_CUTOFF = datetime.date(2025, 12, 15)
END_DATE = min(datetime.date.today(), END_DATE_CUTOFF)
investment_horizon = "M"  # 只允许 'M' 或 'W'
TRAINING_YEARS = 3
OUTLIER_ABS_THRESHOLD = 101  # clamp extremely large absolute factor values before scaling
RARE_VALUE_THRESHOLD = 20  # treat low-cardinality factors as categorical-like for normalization
TRANSACTION_COST_RATE = 0.001  # per turnover cost assumption
PORTFOLIO_SIZE = 50  # target number of holdings
CLEAN_SIMULATION_FILE = True
INDEX_CODE = "000852.XSHG"
LOG_EPS = 1e-6
FACTOR_CATEGORIES = ["risk", "basics"]
_SORTED_TRADE_DATES = None
_SELECTED_FACTORS = None

if investment_horizon not in ("M", "W"):
    raise ValueError("investment_horizon must be 'M' or 'W'")

SIMULATION_FILE = "L10_temp_fixed_m_basicsrisk.pkl"

# 重新设计模型时清理断点续跑文件，防止复用旧状态
if CLEAN_SIMULATION_FILE and os.path.exists(SIMULATION_FILE):
    os.remove(SIMULATION_FILE)


# -----------------------------
# 辅助函数
# -----------------------------
def get_sorted_trade_days():
    global _SORTED_TRADE_DATES
    if _SORTED_TRADE_DATES is None:
        _SORTED_TRADE_DATES = sorted(get_all_trade_days())
    return _SORTED_TRADE_DATES


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
    stacked = all_data.set_index(["time", "code"]).unstack()
    volume = stacked.loc[:, "volume"].replace(0, np.nan)
    money = stacked.loc[:, "money"]
    vwap_prices = (money / volume).replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any")
    if vwap_prices.empty:
        return pd.Series(dtype="float64")

    base_prices = vwap_prices.iloc[0].replace(0, np.nan)
    valid_cols = base_prices[base_prices.notna() & (base_prices != 0)].index
    vwap_prices = vwap_prices.loc[:, valid_cols]
    if vwap_prices.empty:
        return pd.Series(dtype="float64")

    vwap_ret_series = vwap_prices.iloc[-1] / vwap_prices.iloc[0] - 1
    vwap_ret_series = vwap_ret_series.replace([np.inf, -np.inf], np.nan).dropna()
    return vwap_ret_series


def cal_portfolio_vwap_ret(old_portfolio_weight_series, new_portfolio_weight_series, buy_date, sell_date):
    order_book_ids = new_portfolio_weight_series.index.tolist()

    vwap_ret_series = cal_vwap_ret_series(order_book_ids, buy_date, sell_date)
    hpr = (new_portfolio_weight_series * vwap_ret_series).sum()

    old_new = pd.concat([old_portfolio_weight_series, new_portfolio_weight_series], axis=1).fillna(0)
    cost = (old_new.iloc[:, 0] - old_new.iloc[:, 1]).abs().sum() * TRANSACTION_COST_RATE

    return hpr - cost


def get_previous_trade_date(current_date):
    trading_dates = get_sorted_trade_days()
    for trading_date in reversed(trading_dates):
        if trading_date < current_date:
            return trading_date
    return None


def get_next_trade_date(current_date):
    trading_dates = get_sorted_trade_days()
    for trading_date in trading_dates:
        if trading_date > current_date:
            return trading_date
    return None


def get_buy_dates(start_date, end_date, freq: str) -> list:
    periodic_dates = [x.date() for x in pd.date_range(start_date, end_date, freq=freq)]
    return np.sort(
        np.unique([get_next_trade_date(d) for d in periodic_dates if (get_next_trade_date(d) <= end_date)])
    ).tolist()


def normalize_series(series):
    series = series.copy().replace([np.inf, -np.inf], np.nan)

    if series.abs().max() > OUTLIER_ABS_THRESHOLD:
        # 对称取log，既压缩极端值又保留因子方向
        series = np.sign(series) * np.log2(1.0 + series.abs() + LOG_EPS)

    if np.isnan(series.mean()) or np.isnan(series.std()) or (series.std() < 0.000001):
        series.iloc[:] = 0.0
        return series

    if len(series.unique()) <= RARE_VALUE_THRESHOLD:
        series = (series - series.mean()) / series.std()
        return series.fillna(0)

    q = series.quantile([0.01, 0.99])
    series[series < q.iloc[0]] = q.iloc[0]
    series[series > q.iloc[1]] = q.iloc[1]

    series = (series - series.mean()) / series.std()

    series = series.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return series


def get_selected_factors():
    global _SELECTED_FACTORS
    if _SELECTED_FACTORS is None:
        factors = get_all_factors()
        factor_mask = factors.loc[:, "category"].isin(FACTOR_CATEGORIES)
        _SELECTED_FACTORS = factors.loc[factor_mask, "factor"].tolist()
    return _SELECTED_FACTORS


def get_my_factors(decision_date, all_stocks):
    factor_df_list = []
    selected_factors = get_selected_factors()
    for i_factor in selected_factors:
        factor_df_list.append(
            get_factor_values(
                securities=all_stocks,
                factors=i_factor,
                start_date=decision_date,
                end_date=decision_date,
            )[i_factor].T
        )
    factor_df = pd.concat(factor_df_list, axis=1)
    factor_df.columns = selected_factors
    return factor_df


my_model = None


def train_model():
    global my_model
    if my_model is not None:
        return my_model

    training_cutoff_date = get_previous_trade_date(start_date)
    if training_cutoff_date is None:
        raise ValueError(f"Unable to determine a trading date before start_date {start_date} for training data split.")

    training_dates = get_buy_dates(
        start_date=start_date - datetime.timedelta(days=365 * TRAINING_YEARS),
        end_date=training_cutoff_date,
        freq=investment_horizon,
    )

    factor_df_list = []
    for i in tqdm(range(len(training_dates) - 1)):
        buy_date = training_dates[i]
        sell_date = training_dates[i + 1]
        i_pre_date = get_previous_trade_date(buy_date)
        all_stocks = get_index_stocks(INDEX_CODE, date=i_pre_date)
        all_stocks = list(set(all_stocks) - get_st_or_paused_stock_set(i_pre_date))

        factor_df = get_my_factors(i_pre_date, all_stocks)
        factor_df.loc[:, "next_vwap_ret"] = cal_vwap_ret_series(all_stocks, buy_date, sell_date)
        factor_df = factor_df.apply(normalize_series)
        factor_df_list.append(factor_df)

    combined_factor_df = pd.concat(factor_df_list)
    combined_factor_df = combined_factor_df.dropna(subset=["next_vwap_ret"]).fillna(0)

    my_model = Ridge()
    my_model.fit(combined_factor_df.iloc[:, :-1], combined_factor_df.iloc[:, -1])
    return my_model


# -----------------------------
# 投组构建逻辑（保持与 L10 一致）
# -----------------------------
def cal_portfolio_weight_series(decision_date, old_portfolio_weight_series):
    # 在decision_date收盘后决定接下来下一天要买的投资组合
    # 这个函数里面不能使用任何decision_date所在时间之后的信息
    #（decision_date当天收盘的信息依然可用，假设在decision_date的下一天才调仓）
    model = train_model()

    # 基础股票池，去掉st和停牌股
    all_stocks = get_index_stocks(INDEX_CODE, date=decision_date)
    all_stocks = list(set(all_stocks) - get_st_or_paused_stock_set(decision_date))

    # 抓取因子
    factor_df = get_my_factors(decision_date, all_stocks)
    factor_df = factor_df.apply(normalize_series)

    # 预测
    predicted_factor = pd.Series(model.predict(factor_df), index=factor_df.index)

    # 筛选
    filtered_assets = predicted_factor.nlargest(PORTFOLIO_SIZE).index.tolist()

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

        allocation_dict[buy_date] = new_portfolio_weight_series.copy()  # copy to prevent later mutations from altering history

        wealth_process.loc[sell_date] = wealth_process.loc[buy_date] * (
            1 + cal_portfolio_vwap_ret(old_portfolio_weight_series, new_portfolio_weight_series, buy_date, sell_date)
        )

        old_portfolio_weight_series = new_portfolio_weight_series

    return wealth_process, allocation_dict


if __name__ == "__main__":
    wealth_process, allocation_dict = simulate_wealth_process(start_date, END_DATE)
    print(wealth_process.dropna().tail())
