"""
做市研究入口：CSV / 合成数据 → 特征 → XGBoost（时间切分评估 + 全样本回测拟合）→ 回测。
运行：python run.py
"""

from __future__ import annotations

import argparse
import sys
import traceback

import numpy as np
import pandas as pd

from alpha import XGBoostAlphaConfig, XGBoostAlphaModel, build_ml_feature_matrix
from backtest import BacktestConfig, BacktestResult, run_backtest
from engine import ASConfig, AvellanedaStoikovEngine
from features import rolling_volatility


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def load_orderbook(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _normalize_columns(df)


def load_trades(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _normalize_columns(df)


def build_market_from_raw(orderbook: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    ob = orderbook.copy()
    tr = trades.copy()

    for c in ("timestamp",):
        if c not in ob.columns or c not in tr.columns:
            raise ValueError("orderbook/trades must contain 'timestamp' column")

    ob["timestamp"] = ob["timestamp"].astype("int64")
    tr["timestamp"] = tr["timestamp"].astype("int64")
    ob = ob.sort_values("timestamp").reset_index(drop=True)
    tr = tr.sort_values("timestamp").reset_index(drop=True)

    snap = ob[["timestamp"]].rename(columns={"timestamp": "snap_ts"})
    tr_al = pd.merge_asof(
        tr,
        snap,
        left_on="timestamp",
        right_on="snap_ts",
        direction="nearest",
        tolerance=500,
    )

    if tr_al["snap_ts"].notna().any():
        grouped = tr_al.dropna(subset=["snap_ts"]).groupby("snap_ts")
        agg = grouped.agg(
            trade_count=("price", "size"),
            trade_volume=("quantity", "sum"),
            buy_volume=(
                "quantity",
                lambda x: x[tr_al.loc[x.index, "side"] == "buy"].sum(),
            ),
            sell_volume=(
                "quantity",
                lambda x: x[tr_al.loc[x.index, "side"] == "sell"].sum(),
            ),
            last_trade_price=("price", "last"),
        )
        agg = agg.reindex(snap["snap_ts"]).fillna(0.0)
    else:
        agg = pd.DataFrame(
            0.0,
            index=snap["snap_ts"],
            columns=["trade_count", "trade_volume", "buy_volume", "sell_volume", "last_trade_price"],
        )

    market = ob.copy()
    market["bid_price"] = market["bid_price_1"].astype(float)
    market["ask_price"] = market["ask_price_1"].astype(float)
    market["bid_size"] = market["bid_volume_1"].astype(float)
    market["ask_size"] = market["ask_volume_1"].astype(float)
    market["volume"] = agg["trade_volume"].to_numpy(dtype=float)

    if "timestamp" in market.columns:
        market = market.sort_values("timestamp").reset_index(drop=True)

    return market


def make_sigma_series(mid: pd.Series, window: int) -> pd.Series:
    v = rolling_volatility(mid, window=window, returns="log")
    return v.bfill().ffill()


def _print_alpha_eval(report: dict) -> None:
    print("--- Alpha 评估（时间顺序 hold-out）---")
    print(f"IC (Pearson):     {report.get('ic_pearson', float('nan')):.6f}")
    print(f"IC (Spearman):    {report.get('ic_spearman', float('nan')):.6f}")
    print(f"方向命中率:       {report.get('accuracy_direction', float('nan')):.6f}")
    print(f"分类准确率(测):   {report.get('classification_accuracy', float('nan')):.6f}")
    print(f"训练行数:         {int(report.get('train_rows', 0))}")
    print(f"测试行数:         {int(report.get('test_rows', 0))}")
    br = report.get("bucket_mean_return") or {}
    if br:
        print("分层收益(未来对数收益均值):")
        for k in sorted(br.keys()):
            print(f"  {k}: {br[k]:.8f}")


def run_pipeline(
    market: pd.DataFrame,
    as_config: ASConfig,
    backtest_config: BacktestConfig | None = None,
    vol_window: int = 20,
    train_alpha: bool = True,
    alpha_cfg: XGBoostAlphaConfig | None = None,
) -> tuple[BacktestResult, pd.DataFrame, pd.Series | None, dict | None]:
    if backtest_config is None:
        backtest_config = BacktestConfig(volatility_window=vol_window)

    book_need = {"bid_price", "ask_price", "bid_size", "ask_size"}
    if not book_need.issubset(market.columns):
        print(
            "警告: 缺少完整订单簿列；跳过 ML alpha。",
            file=sys.stderr,
        )

    if "mid" not in market.columns and "bid_price" in market.columns:
        mid = (market["bid_price"].astype(float) + market["ask_price"].astype(float)) * 0.5
    else:
        mid = market["mid"].astype(float)

    sigma = make_sigma_series(mid, window=vol_window)

    alpha_series: pd.Series | None = None
    eval_report: dict | None = None

    if train_alpha and book_need.issubset(market.columns):
        try:
            ac = alpha_cfg or XGBoostAlphaConfig(vol_window=vol_window)
            mdl = XGBoostAlphaModel(ac)
            _, eval_report = mdl.fit_time_split(market, config=ac)
            mdl.fit_full(market, config=ac)
            alpha_series = mdl.predict_alpha_score(market)
        except Exception:
            print("Alpha 流水线失败，使用 NaN alpha（回测不滤信号）。", file=sys.stderr)
            traceback.print_exc()
            alpha_series = pd.Series(np.nan, index=market.index, name="alpha")
            eval_report = None

    mbt = market.copy()
    if alpha_series is not None:
        mbt["alpha"] = alpha_series.reindex(market.index)
    else:
        mbt["alpha"] = 0.0

    if "imbalance" not in mbt.columns:
        mbt["imbalance"] = 0.0

    feature_df = (
        build_ml_feature_matrix(market)
        if book_need.issubset(market.columns)
        else pd.DataFrame(index=market.index)
    )

    engine = AvellanedaStoikovEngine(as_config)
    result = run_backtest(engine, mbt, sigma=sigma, config=backtest_config)
    return result, feature_df, alpha_series, eval_report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="做市研究：alpha 评估 + AS 回测")
    p.add_argument("--orderbook", type=str, default="orderbook.csv", help="orderbook.csv path")
    p.add_argument("--trades", type=str, default="trades.csv", help="trades.csv path")

    p.add_argument("--gamma", type=float, default=0.02)
    p.add_argument("--kappa", type=float, default=5.0)
    p.add_argument("--sigma", type=float, default=0.01)
    p.add_argument("--time-horizon", type=float, default=1.0)
    p.add_argument("--spread-scale", type=float, default=0.28)
    p.add_argument("--gamma-inventory", type=float, default=0.018, help="保留价库存项系数")

    p.add_argument("--quote-size", type=float, default=1.0)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--latency-bars", type=int, default=0)
    p.add_argument("--max-inventory", type=float, default=4.0)
    p.add_argument("--min-half-spread", type=float, default=0.0)
    p.add_argument("--vol-window", type=int, default=20)

    p.add_argument("--fill-seed", type=int, default=42)
    p.add_argument("--fill-prob", type=float, default=0.45)
    p.add_argument("--quote-slip-bps", type=float, default=0.0)
    p.add_argument("--fill-slip-bps", type=float, default=1.2)

    p.add_argument("--alpha-min-abs", type=float, default=0.2)
    p.add_argument("--disable-alpha-filter", action="store_true")
    p.add_argument("--no-train-alpha", action="store_true")

    p.add_argument("--horizon-decay", action="store_true")

    return p.parse_args(argv)


def _summarize_bt(result: BacktestResult) -> None:
    s = result.summary
    lines = [
        "--- 回测摘要 ---",
        f"PnL (MTM):       {s['pnl']:.6f}",
        f"初始权益:         {s['initial_equity']:.6f}",
        f"期末权益:         {s['final_equity']:.6f}",
        f"手续费合计:       {s['realized_fees']:.6f}",
        f"期末现金:         {s['final_cash']:.6f}",
        f"期末库存:         {s['final_inventory']:.6f}",
        f"库存变化:         {s.get('inventory_delta', 0.0):.6f}",
        f"成交笔数:         {int(s.get('n_trades', 0))}",
        f"买/卖笔数:        {int(s['n_buys'])} / {int(s['n_sells'])}",
        f"状态行数:         {len(result.states)}",
    ]
    print("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    ob = load_orderbook(args.orderbook)
    tr = load_trades(args.trades)
    market = build_market_from_raw(ob, tr)

    need = "mid" in market.columns or (
        "bid_price" in market.columns and "ask_price" in market.columns
    )
    if not need:
        print("数据需包含 mid 或 bid_price+ask_price。", file=sys.stderr)
        return 1

    as_cfg = ASConfig(
        gamma=args.gamma,
        kappa=args.kappa,
        sigma=args.sigma,
        time_horizon=args.time_horizon,
        spread_scale=args.spread_scale,
        gamma_inventory=args.gamma_inventory,
    )
    bt_cfg = BacktestConfig(
        quote_size=args.quote_size,
        fee_rate=args.fee_rate,
        fee_bps=0.0,
        quote_slippage_bps=args.quote_slip_bps,
        fill_slippage_bps=args.fill_slip_bps,
        latency_bars=args.latency_bars,
        max_inventory=args.max_inventory,
        min_half_spread=args.min_half_spread,
        volatility_window=args.vol_window,
        horizon_decay=args.horizon_decay,
        fill_seed=args.fill_seed,
        fill_realization_prob=args.fill_prob,
        alpha_min_abs=args.alpha_min_abs,
        enable_alpha_filter=not args.disable_alpha_filter,
    )

    alpha_ml_cfg = XGBoostAlphaConfig(
        vol_window=args.vol_window,
        horizon=3,
        lookback=10,
        train_fraction=0.7,
        n_estimators=350,
    )

    result, features, _alpha, ev = run_pipeline(
        market,
        as_config=as_cfg,
        backtest_config=bt_cfg,
        vol_window=args.vol_window,
        train_alpha=not args.no_train_alpha,
        alpha_cfg=alpha_ml_cfg,
    )

    print(f"行情行数: {len(market)} | 特征列数: {features.shape[1]}")
    if ev is not None:
        _print_alpha_eval(ev)
    print()
    _summarize_bt(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
