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


def _pick_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").astype(float)
    return pd.Series(np.nan, index=df.index, dtype=float)


def level_ofi(
    df: pd.DataFrame,
    level: int,
    noise_threshold: float = 0.0,
) -> pd.Series:
    """
    OFI_level = Δbid_volume_level - Δask_volume_level
    """
    b = _pick_col(
        df,
        (f"bid_volume_{level}", f"bid_size_{level}", f"bid_vol_{level}"),
    )
    a = _pick_col(
        df,
        (f"ask_volume_{level}", f"ask_size_{level}", f"ask_vol_{level}"),
    )
    db = b.diff()
    da = a.diff()
    ofi = (db - da).fillna(0.0)
    if noise_threshold > 0:
        mask = (db.abs().fillna(0.0) + da.abs().fillna(0.0)) < noise_threshold
        ofi = ofi.where(~mask, 0.0)
    return ofi.rename(f"ofi_l{level}")


def multi_level_ofi(
    df: pd.DataFrame,
    levels: tuple[int, int, int] = (1, 2, 3),
    weights: tuple[float, float, float] = (0.6, 0.3, 0.1),
    noise_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    L1/L2/L3 OFI + 加权 OFI_total
    """
    if len(levels) != 3 or len(weights) != 3:
        raise ValueError("levels and weights must have length 3")
    w = np.asarray(weights, dtype=float)
    if np.allclose(np.abs(w).sum(), 0.0):
        raise ValueError("weights must not all be zero")
    w = w / np.abs(w).sum()

    l1 = level_ofi(df, levels[0], noise_threshold=noise_threshold)
    l2 = level_ofi(df, levels[1], noise_threshold=noise_threshold)
    l3 = level_ofi(df, levels[2], noise_threshold=noise_threshold)
    total = (w[0] * l1 + w[1] * l2 + w[2] * l3).rename("ofi_total")
    return pd.DataFrame(
        {
            "ofi_l1": l1,
            "ofi_l2": l2,
            "ofi_l3": l3,
            "ofi_total": total,
        },
        index=df.index,
    )


def trade_flow(
    buy_volume: pd.Series,
    sell_volume: pd.Series,
) -> pd.Series:
    """
    TradeFlow = buy_volume - sell_volume
    """
    b = pd.to_numeric(buy_volume, errors="coerce").astype(float).fillna(0.0)
    s = pd.to_numeric(sell_volume, errors="coerce").astype(float).fillna(0.0)
    return (b - s).rename("trade_flow")


def estimate_liquidity_k(
    orderbook: pd.DataFrame,
    trades: pd.DataFrame,
    window: int = 500,
    epsilon: float = 1e-6,
    min_k: float = 0.01,
    max_k: float = 100.0,
    tolerance_ms: int = 500,
) -> pd.Series:
    """
    Estimate liquidity parameter K with engineering-MLE approximation:
        K ~= 1 / rolling_mean(delta + epsilon)
    where delta = |trade_price - mid_price|.
    """
    if orderbook.empty:
        return pd.Series(dtype=float, name="liquidity_k")

    ob = orderbook.copy()
    tr = trades.copy() if trades is not None else pd.DataFrame()
    ob.columns = [str(c).strip().lower() for c in ob.columns]
    if not tr.empty:
        tr.columns = [str(c).strip().lower() for c in tr.columns]

    if "timestamp" not in ob.columns:
        raise ValueError("orderbook must include timestamp")
    if "bid_price_1" not in ob.columns or "ask_price_1" not in ob.columns:
        raise ValueError("orderbook must include bid_price_1/ask_price_1")

    snap = ob[["timestamp"]].copy()
    snap["timestamp"] = pd.to_numeric(snap["timestamp"], errors="coerce").astype("int64")
    mid = (
        pd.to_numeric(ob["bid_price_1"], errors="coerce")
        + pd.to_numeric(ob["ask_price_1"], errors="coerce")
    ) * 0.5
    snap["mid"] = mid.astype(float)
    snap = snap.sort_values("timestamp").reset_index(drop=True)

    if tr.empty or "timestamp" not in tr.columns or "price" not in tr.columns:
        base = np.full(len(snap), float(np.clip(1.0 / epsilon, min_k, max_k)), dtype=float)
        return pd.Series(base, index=orderbook.index, name="liquidity_k")

    ta = tr[["timestamp", "price"]].copy()
    ta["timestamp"] = pd.to_numeric(ta["timestamp"], errors="coerce")
    ta["price"] = pd.to_numeric(ta["price"], errors="coerce")
    ta = ta.dropna(subset=["timestamp", "price"]).astype({"timestamp": "int64"})
    ta = ta.sort_values("timestamp").reset_index(drop=True)
    if ta.empty:
        base = np.full(len(snap), float(np.clip(1.0 / epsilon, min_k, max_k)), dtype=float)
        return pd.Series(base, index=orderbook.index, name="liquidity_k")

    mapped = pd.merge_asof(
        ta,
        snap[["timestamp", "mid"]].rename(columns={"timestamp": "snap_ts"}),
        left_on="timestamp",
        right_on="snap_ts",
        direction="nearest",
        tolerance=tolerance_ms,
    )
    mapped = mapped.dropna(subset=["snap_ts", "mid"]).copy()
    if mapped.empty:
        base = np.full(len(snap), float(np.clip(1.0 / epsilon, min_k, max_k)), dtype=float)
        return pd.Series(base, index=orderbook.index, name="liquidity_k")

    mapped["delta"] = (mapped["price"] - mapped["mid"]).abs().astype(float)
    delta_mean = (
        mapped.groupby("snap_ts")["delta"]
        .mean()
        .reindex(snap["timestamp"])
        .ffill()
        .fillna(0.0)
    )
    roll = delta_mean.rolling(window=window, min_periods=max(10, window // 10)).mean()
    roll = roll.ffill().fillna(delta_mean + epsilon)
    k = 1.0 / (roll + float(epsilon))
    k = k.clip(lower=min_k, upper=max_k)
    return pd.Series(k.to_numpy(dtype=float), index=orderbook.index, name="liquidity_k")
