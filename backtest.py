"""
回测：订单簿交叉成交、成交概率、部分成交、简单队列、手续费与滑点、MTM PnL。
与 ``engine.AvellanedaStoikovEngine`` 集成；报价使用 alpha 参与 reservation。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from engine import AvellanedaStoikovEngine


def _ensure_mid_ohlc(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    if "mid" in df.columns:
        mid = df["mid"].astype(float)
    elif "bid_price" in df.columns and "ask_price" in df.columns:
        mid = (df["bid_price"].astype(float) + df["ask_price"].astype(float)) * 0.5
    else:
        raise ValueError("需要列 'mid' 或同时有 'bid_price' 与 'ask_price'")

    if "low" in df.columns:
        low = df["low"].astype(float)
    else:
        low = mid.copy()
    if "high" in df.columns:
        high = df["high"].astype(float)
    else:
        high = mid.copy()
    return mid, low, high


def _market_bid_ask(
    market: pd.DataFrame,
    mid: pd.Series,
    i: int,
    default_spread_bps: float,
) -> tuple[float, float]:
    if "bid_price" in market.columns and "ask_price" in market.columns:
        mb = float(market["bid_price"].iloc[i])
        ma = float(market["ask_price"].iloc[i])
        return mb, ma
    sm = float(mid.iloc[i])
    half = sm * default_spread_bps / 1e4 * 0.5
    return sm - half, sm + half


def _rolling_sigma(mid: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    if min_periods is None:
        min_periods = max(2, window // 2)
    lr = np.log(mid.astype(float)).diff()
    return lr.rolling(window=window, min_periods=min_periods).std()


@dataclass
class BacktestConfig:
    quote_size: float = 1.0
    fee_bps: float = 0.0
    fee_rate: float = 0.0004
    quote_slippage_bps: float = 0.0
    fill_slippage_bps: float = 1.2
    latency_bars: int = 0
    volatility_window: int = 20
    max_inventory: float | None = None
    min_half_spread: float = 0.0
    initial_cash: float = 0.0
    initial_inventory: float = 0.0

    default_spread_bps: float = 8.0
    """当无 bid/ask 时，用 mid±half 构造价差（半宽为 spread_bps/2）。"""

    horizon_decay: bool = False
    extra_engine_kwargs: dict[str, Any] = field(default_factory=dict)

    alpha_imbalance_col: str = "imbalance"
    cross_imbalance_threshold: float = 0.25
    cross_bps: float = 12.0

    fill_realization_prob: float = 0.45
    """交叉后实际发生成交的 Bernoulli 概率（建议 0.3–0.6）。"""

    fill_prob_min: float = 0.30
    fill_prob_max: float = 0.60
    """若 >0 且与 fill_realization_prob 联用：每 bar 在区间内微扰（可选）。"""

    partial_fill_min: float = 0.35
    partial_fill_max: float = 1.0

    queue_initial_scale: float = 0.85
    queue_drain_scale: float = 0.22
    """简单队列：交叉后先消耗队列深度再允许成交。"""

    fill_seed: int = 42
    one_fill_per_bar: bool = True

    alpha_signal_col: str = "alpha"
    alpha_min_abs: float = 0.2
    enable_alpha_filter: bool = True

    use_book_cross: bool = True
    """True：仅当 bid_quote >= market_ask 或 ask_quote <= market_bid 时可能成交。"""


@dataclass
class BacktestResult:
    states: pd.DataFrame
    trades: pd.DataFrame
    summary: dict[str, float]


def run_backtest(
    engine: AvellanedaStoikovEngine,
    market: pd.DataFrame,
    sigma: pd.Series | None = None,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    if config is None:
        config = BacktestConfig()

    if config.quote_size <= 0:
        raise ValueError("quote_size 必须为正")
    if config.latency_bars < 0:
        raise ValueError("latency_bars 必须非负")
    if config.fee_rate < 0 or config.fee_bps < 0:
        raise ValueError("fee 必须非负")

    def trade_fee(notional: float) -> float:
        r = config.fee_rate if config.fee_rate > 0.0 else config.fee_bps / 1e4
        return notional * r

    mid, low, high = _ensure_mid_ohlc(market)
    n = len(market.index)

    if sigma is None and "sigma" in market.columns:
        sigma_series = market["sigma"].astype(float)
    elif sigma is not None:
        sigma_series = sigma.reindex(market.index).astype(float)
    else:
        sigma_series = _rolling_sigma(mid, window=config.volatility_window)
        sigma_series = sigma_series.bfill().ffill()
        fallback = float(mid.pct_change().std()) if n > 1 else 1e-6
        sigma_series = sigma_series.fillna(fallback)
        sigma_series = sigma_series.replace(0.0, fallback)

    cash = float(config.initial_cash)
    inv = float(config.initial_inventory)

    cfg = engine.config
    orig_tau = float(cfg.time_horizon)
    orig_sigma = float(cfg.sigma)
    rows_state: list[dict[str, Any]] = []
    rows_trade: list[dict[str, Any]] = []

    bid_work = np.full(n, np.nan)
    ask_work = np.full(n, np.nan)
    queue_bid_snapshot = np.zeros(n)
    queue_ask_snapshot = np.zeros(n)

    engine.update_horizon(orig_tau)
    rng = np.random.default_rng(config.fill_seed)

    same_bar = config.latency_bars == 0

    try:
        for i in range(n):
            idx = market.index[i]
            s_mid = float(mid.iloc[i])
            sig = float(sigma_series.iloc[i])
            if not np.isfinite(sig) or sig <= 0:
                sig = 1e-8

            if config.horizon_decay and n > 1:
                tau_scale = 1.0 - i / (n - 1)
                engine.update_horizon(max(orig_tau * tau_scale, 1e-8))

            cfg.sigma = sig

            alpha_val = 0.0
            alpha_raw: float | None = None
            if config.alpha_signal_col in market.columns:
                try:
                    av = market[config.alpha_signal_col].iloc[i]
                    if pd.isna(av):
                        alpha_raw = float("nan")
                    else:
                        alpha_val = float(av)
                        alpha_raw = alpha_val
                except (TypeError, ValueError):
                    alpha_raw = None
                    alpha_val = 0.0

            alpha_ok = True
            if config.enable_alpha_filter and config.alpha_signal_col in market.columns:
                if alpha_raw is not None and isinstance(alpha_raw, float) and np.isnan(alpha_raw):
                    alpha_ok = True
                else:
                    alpha_ok = np.isfinite(alpha_val) and abs(alpha_val) >= config.alpha_min_abs

            q_kwargs: dict[str, Any] = {
                "min_half_spread": config.min_half_spread,
                "max_inventory": config.max_inventory,
                "alpha": alpha_val if np.isfinite(alpha_val) else None,
            }
            q_kwargs.update(config.extra_engine_kwargs)
            bid_q, ask_q = engine.quote_prices_clamped(s_mid, inv, **q_kwargs)

            qsb = config.quote_slippage_bps / 1e4
            bid_q = float(bid_q) * (1.0 - qsb)
            ask_q = float(ask_q) * (1.0 + qsb)

            imb = 0.0
            if config.alpha_imbalance_col in market.columns:
                try:
                    imb = float(market[config.alpha_imbalance_col].iloc[i])
                except (TypeError, ValueError):
                    imb = 0.0
                if not np.isfinite(imb):
                    imb = 0.0
            thr = config.cross_imbalance_threshold
            cb = config.cross_bps / 1e4
            if imb > thr:
                w = min(1.0, abs(imb))
                bid_q = max(bid_q, s_mid * (1.0 + cb * w))
            if imb < -thr:
                w = min(1.0, abs(imb))
                ask_q = min(ask_q, s_mid * (1.0 - cb * w))
            if bid_q >= ask_q:
                eps = 1e-6 * max(s_mid, 1e-12)
                bid_q = s_mid - eps
                ask_q = s_mid + eps

            if not alpha_ok:
                bid_q = float("nan")
                ask_q = float("nan")

            if same_bar:
                b, a = bid_q, ask_q
            else:
                b = bid_work[i]
                a = ask_work[i]
                if not (np.isfinite(b) and np.isfinite(a)):
                    b = float("nan")
                    a = float("nan")

            m_bid, m_ask = _market_bid_ask(market, mid, i, config.default_spread_bps)
            fsb = config.fill_slippage_bps / 1e4
            cap = config.max_inventory

            p_fill = float(config.fill_realization_prob)
            if config.fill_prob_max > config.fill_prob_min:
                p_fill = float(
                    rng.uniform(config.fill_prob_min, config.fill_prob_max)
                )

            if same_bar and np.isfinite(bid_q) and np.isfinite(ask_q) and alpha_ok:
                # 同 bar 撮合：队列已在簿前；随机性由 fill_realization_prob 承担
                qb_rem = 0.0
                qa_rem = 0.0
            else:
                qb_rem = float(queue_bid_snapshot[i])
                qa_rem = float(queue_ask_snapshot[i])

            if config.use_book_cross and np.isfinite(b) and np.isfinite(a) and alpha_ok:
                bid_cross = float(b) >= m_ask
                ask_cross = float(a) <= m_bid

                if bid_cross:
                    qb_rem -= (
                        rng.uniform(0.12, 0.35)
                        * config.quote_size
                        * config.queue_drain_scale
                    )
                if ask_cross:
                    qa_rem -= (
                        rng.uniform(0.12, 0.35)
                        * config.quote_size
                        * config.queue_drain_scale
                    )
                qb_rem = float(max(qb_rem, 0.0))
                qa_rem = float(max(qa_rem, 0.0))

                can_buy = cap is None or inv < cap
                can_sell = cap is None or inv > -cap

                imb_bar = 0.0
                if config.alpha_imbalance_col in market.columns:
                    try:
                        imb_bar = float(market[config.alpha_imbalance_col].iloc[i])
                    except (TypeError, ValueError):
                        imb_bar = 0.0
                    if not np.isfinite(imb_bar):
                        imb_bar = 0.0

                do_buy = can_buy and bid_cross and qb_rem <= 0
                do_sell = can_sell and ask_cross and qa_rem <= 0

                if config.one_fill_per_bar and do_buy and do_sell:
                    if imb_bar >= 0.0:
                        do_sell = False
                    else:
                        do_buy = False

                if do_buy:
                    do_buy = rng.random() < p_fill
                if do_sell:
                    do_sell = rng.random() < p_fill

                frac_b = rng.uniform(config.partial_fill_min, config.partial_fill_max)
                frac_s = rng.uniform(config.partial_fill_min, config.partial_fill_max)
                q_buy = config.quote_size * frac_b
                q_sell = config.quote_size * frac_s

                if do_buy:
                    fill_px = m_ask * (1.0 + fsb)
                    notional = fill_px * q_buy
                    fee = trade_fee(notional)
                    cash -= notional + fee
                    inv += q_buy
                    rows_trade.append(
                        {
                            "timestamp": idx,
                            "side": "buy",
                            "qty": q_buy,
                            "price": fill_px,
                            "fee": fee,
                            "inventory_after": inv,
                            "cash_after": cash,
                        }
                    )

                can_sell = cap is None or inv > -cap
                if can_sell and do_sell and not (config.one_fill_per_bar and do_buy):
                    fill_px = m_bid * (1.0 - fsb)
                    notional = fill_px * q_sell
                    fee = trade_fee(notional)
                    cash += notional - fee
                    inv -= q_sell
                    rows_trade.append(
                        {
                            "timestamp": idx,
                            "side": "sell",
                            "qty": q_sell,
                            "price": fill_px,
                            "fee": fee,
                            "inventory_after": inv,
                            "cash_after": cash,
                        }
                    )

                queue_bid_snapshot[i] = qb_rem
                queue_ask_snapshot[i] = qa_rem

            if not same_bar:
                eff_i = i + 1 + config.latency_bars
                if eff_i < n and alpha_ok and np.isfinite(bid_q) and np.isfinite(ask_q):
                    bid_work[eff_i] = bid_q
                    ask_work[eff_i] = ask_q
                    queue_bid_snapshot[eff_i] = (
                        rng.uniform(0.25, 1.0)
                        * config.quote_size
                        * config.queue_initial_scale
                    )
                    queue_ask_snapshot[eff_i] = (
                        rng.uniform(0.25, 1.0)
                        * config.quote_size
                        * config.queue_initial_scale
                    )

            equity = cash + inv * s_mid
            row_state: dict[str, Any] = {
                "timestamp": idx,
                "mid": s_mid,
                "sigma": sig,
                "bid_quote": b,
                "ask_quote": a,
                "inventory": inv,
                "cash": cash,
                "equity": equity,
                "mtm_pnl": equity - (config.initial_cash + config.initial_inventory * s_mid),
            }
            if config.alpha_signal_col in market.columns:
                row_state["alpha_signal"] = alpha_val
                row_state["alpha_trade_ok"] = bool(alpha_ok)
            rows_state.append(row_state)
    finally:
        engine.update_horizon(orig_tau)
        cfg.sigma = orig_sigma

    states = pd.DataFrame(rows_state)
    trades = pd.DataFrame(rows_trade)

    realized_fees = float(trades["fee"].sum()) if len(trades) else 0.0
    final_mid = float(mid.iloc[-1])
    mid0 = float(mid.iloc[0])
    initial_equity = config.initial_cash + config.initial_inventory * mid0
    final_equity = cash + inv * final_mid
    pnl_net = float(final_equity - initial_equity)
    gross_pnl = float(pnl_net + realized_fees)

    if len(trades) > 0:
        n_buys = float((trades["side"] == "buy").sum())
        n_sells = float((trades["side"] == "sell").sum())
        n_trades = float(len(trades))
    else:
        n_buys = 0.0
        n_sells = 0.0
        n_trades = 0.0

    inv_delta = float(inv - config.initial_inventory)

    summary = {
        "pnl": pnl_net,
        "pnl_net_of_fees": pnl_net,
        "gross_pnl": gross_pnl,
        "total_fees": realized_fees,
        "realized_fees": realized_fees,
        "initial_equity": initial_equity,
        "final_cash": cash,
        "final_inventory": inv,
        "inventory_delta": inv_delta,
        "final_equity": final_equity,
        "n_buys": n_buys,
        "n_sells": n_sells,
        "n_trades": n_trades,
    }

    return BacktestResult(states=states, trades=trades, summary=summary)


def synthetic_ohlc(
    n: int = 500,
    seed: int = 0,
    dt_vol: float = 0.0018,
    drift: float = 0.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = drift + rng.normal(0.0, dt_vol, size=n)
    log_mid = np.cumsum(r)
    mid = 100.0 * np.exp(log_mid)
    noise = rng.uniform(0.08, 0.45, size=n) * np.abs(r) * mid + np.abs(
        rng.normal(0, 0.55 * dt_vol * mid, size=n)
    )
    high = mid + noise + np.abs(rng.normal(0, 0.85 * dt_vol * mid, size=n))
    low = mid - noise - np.abs(rng.normal(0, 0.85 * dt_vol * mid, size=n))
    imbalance = np.clip(rng.normal(0.0, 0.65, size=n), -1.0, 1.0)
    idx = pd.RangeIndex(n)
    return pd.DataFrame(
        {"mid": mid, "high": high, "low": low, "imbalance": imbalance}, index=idx
    )


def synthetic_order_book(
    n: int = 800,
    seed: int = 0,
    dt_vol: float = 0.0016,
    base_spread_bps: float = 7.0,
) -> pd.DataFrame:
    """合成限价簿：bid/ask、量、成交量，用于 ML 与回测。"""
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, dt_vol, size=n)
    log_mid = np.cumsum(r)
    mid = 100.0 * np.exp(log_mid)
    half = mid * (base_spread_bps / 1e4) * 0.5 * rng.uniform(0.85, 1.15, size=n)
    bp = mid - half
    ap = mid + half
    bs = rng.lognormal(mean=1.4, sigma=0.35, size=n)
    qs = rng.lognormal(mean=1.4, sigma=0.35, size=n)
    vol = np.maximum(0.0, (bs + qs) * rng.uniform(0.08, 0.32, size=n))
    imbalance = np.clip(rng.normal(0.0, 0.55, size=n), -1.0, 1.0)
    idx = pd.RangeIndex(n)
    return pd.DataFrame(
        {
            "bid_price": bp,
            "ask_price": ap,
            "bid_size": bs,
            "ask_size": qs,
            "volume": vol,
            "imbalance": imbalance,
        },
        index=idx,
    )
