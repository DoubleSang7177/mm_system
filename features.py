"""
特征工程：滚动统计、OFI、成交量 surge、bid/ask delta 等（仅使用过去信息）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    p = prices.astype(float)
    return np.log(p).diff(periods).rename(f"log_ret_{periods}")


def pct_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    p = prices.astype(float)
    return (p / p.shift(periods) - 1.0).rename(f"pct_ret_{periods}")


def rolling_mean(
    s: pd.Series, window: int, min_periods: int | None = None
) -> pd.Series:
    if min_periods is None:
        min_periods = window
    return s.astype(float).rolling(window=window, min_periods=min_periods).mean()


def rolling_std(
    s: pd.Series,
    window: int,
    min_periods: int | None = None,
    ddof: int = 1,
) -> pd.Series:
    if min_periods is None:
        min_periods = window
    return s.astype(float).rolling(window=window, min_periods=min_periods).std(ddof=ddof)


def rolling_momentum(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """过去 window 步的价格变化（末 - window 前）。"""
    if min_periods is None:
        min_periods = window
    x = s.astype(float)
    return (x - x.shift(window)).rename(f"mom_{window}")


def rolling_volatility(
    prices: pd.Series,
    window: int,
    returns: str = "log",
    annualize: float | None = None,
    min_periods: int | None = None,
) -> pd.Series:
    if returns not in ("log", "simple"):
        raise ValueError('returns must be "log" or "simple"')
    if min_periods is None:
        min_periods = window
    r = log_returns(prices, 1) if returns == "log" else pct_returns(prices, 1)
    vol = rolling_std(r, window=window, min_periods=min_periods)
    if annualize is not None:
        vol = vol * np.sqrt(annualize)
    return vol.rename(f"vol_{returns}_w{window}")


def rolling_zscore(
    s: pd.Series, window: int, min_periods: int | None = None
) -> pd.Series:
    if min_periods is None:
        min_periods = window
    x = s.astype(float)
    m = rolling_mean(x, window=window, min_periods=min_periods)
    sd = rolling_std(x, window=window, min_periods=min_periods)
    return ((x - m) / sd).rename(f"zscore_w{window}")


def order_flow_imbalance(
    bid_price: pd.Series,
    ask_price: pd.Series,
    bid_size: pd.Series,
    ask_size: pd.Series,
) -> pd.Series:
    bp = bid_price.astype(float)
    ap = ask_price.astype(float)
    bs = bid_size.astype(float)
    qs = ask_size.astype(float)
    db = bp.diff().fillna(0.0)
    da = ap.diff().fillna(0.0)
    dbs = bs.diff().fillna(0.0)
    das = qs.diff().fillna(0.0)
    ofi = np.where(db > 0, dbs, np.where(db < 0, -dbs, 0.0))
    ofi = ofi - np.where(da > 0, -das, np.where(da < 0, das, 0.0))
    return pd.Series(ofi, index=bid_price.index, name="ofi")


def volume_surge_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    v = volume.astype(float).clip(lower=0.0)
    ma = v.rolling(window=window, min_periods=max(3, window // 2)).mean()
    return (v / (ma + 1e-12) - 1.0).rename("volume_surge")


def bid_ask_deltas(
    bid_price: pd.Series, ask_price: pd.Series
) -> tuple[pd.Series, pd.Series]:
    db = bid_price.astype(float).diff().rename("delta_bid")
    da = ask_price.astype(float).diff().rename("delta_ask")
    return db, da


def add_rolling_features(
    df: pd.DataFrame,
    columns: dict[str, str],
    windows: list[int],
    include_zscore: bool = False,
) -> pd.DataFrame:
    out = df.copy()
    for prefix, col in columns.items():
        if col not in out.columns:
            raise ValueError(f"missing column {col!r}")
        s = out[col]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        for w in windows:
            out[f"{prefix}_mean_w{w}"] = rolling_mean(s, window=w)
            out[f"{prefix}_std_w{w}"] = rolling_std(s, window=w)
            if include_zscore:
                out[f"{prefix}_zscore_w{w}"] = rolling_zscore(s, window=w)
    return out


def stack_rolling_features(
    series_dict: dict[str, pd.Series],
    window: int,
    prefixes: list[str] | None = None,
) -> pd.DataFrame:
    if prefixes is None:
        prefixes = list(series_dict.keys())
    out = pd.DataFrame(index=next(iter(series_dict.values())).index)
    for name in prefixes:
        s = series_dict[name].astype(float)
        out[f"{name}_rmean_{window}"] = rolling_mean(s, window=window)
        out[f"{name}_rstd_{window}"] = rolling_std(s, window=window)
    return out
