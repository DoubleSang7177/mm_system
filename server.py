from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from features import estimate_liquidity_k

APP_ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = APP_ROOT / "frontend"
ORDERBOOK_PATH = APP_ROOT / "orderbook.csv"
TRADES_PATH = APP_ROOT / "trades.csv"
FEE_RATE = 0.0004
MAX_POINTS = 400

app = FastAPI(title="MM Dashboard API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _prepare_orderbook(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    required = ["timestamp", "bid_price_1", "ask_price_1"]
    if any(c not in df.columns for c in required):
        return pd.DataFrame()
    out = df.copy()
    out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce")
    out["bid_price_1"] = pd.to_numeric(out["bid_price_1"], errors="coerce")
    out["ask_price_1"] = pd.to_numeric(out["ask_price_1"], errors="coerce")
    out["mid"] = (out["bid_price_1"] + out["ask_price_1"]) * 0.5
    out = out.dropna(subset=["timestamp", "mid"])
    out["timestamp"] = out["timestamp"].astype("int64")
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out


def _prepare_trades(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    required = ["timestamp", "price", "quantity", "side"]
    if any(c not in df.columns for c in required):
        return pd.DataFrame()
    out = df.copy()
    out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce")
    out["side"] = out["side"].astype(str).str.lower()
    out = out.dropna(subset=["timestamp", "price", "quantity"])
    out["timestamp"] = out["timestamp"].astype("int64")
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out


def _nearest_align_trades(ob: pd.DataFrame, tr: pd.DataFrame, tolerance_ms: int = 500) -> pd.DataFrame:
    if ob.empty:
        return pd.DataFrame(index=pd.Index([], name="timestamp"))
    snap = ob[["timestamp"]].rename(columns={"timestamp": "snap_ts"})
    if tr.empty:
        return pd.DataFrame(
            0.0,
            index=snap["snap_ts"],
            columns=["trade_count", "trade_volume", "buy_volume", "sell_volume", "last_trade_price"],
        )
    ta = pd.merge_asof(
        tr,
        snap,
        left_on="timestamp",
        right_on="snap_ts",
        direction="nearest",
        tolerance=tolerance_ms,
    )
    if ta["snap_ts"].notna().sum() == 0:
        return pd.DataFrame(
            0.0,
            index=snap["snap_ts"],
            columns=["trade_count", "trade_volume", "buy_volume", "sell_volume", "last_trade_price"],
        )
    mapped = ta.dropna(subset=["snap_ts"]).copy()
    g = mapped.groupby("snap_ts")
    agg = g.agg(
        trade_count=("price", "size"),
        trade_volume=("quantity", "sum"),
        buy_volume=("quantity", lambda x: x[mapped.loc[x.index, "side"] == "buy"].sum()),
        sell_volume=("quantity", lambda x: x[mapped.loc[x.index, "side"] == "sell"].sum()),
        last_trade_price=("price", "last"),
    )
    agg = agg.reindex(snap["snap_ts"]).fillna(0.0)
    return agg


def _compute_alpha_proxy(ob: pd.DataFrame, flow: pd.DataFrame, window: int = 20) -> pd.Series:
    if ob.empty:
        return pd.Series(dtype=float)
    if "bid_volume_1" in ob.columns and "ask_volume_1" in ob.columns:
        bid_v = pd.to_numeric(ob["bid_volume_1"], errors="coerce").fillna(0.0)
        ask_v = pd.to_numeric(ob["ask_volume_1"], errors="coerce").fillna(0.0)
        imbalance = (bid_v - ask_v) / (bid_v + ask_v + 1e-9)
    else:
        imbalance = pd.Series(0.0, index=ob.index)
    signed_flow = (flow["buy_volume"] - flow["sell_volume"]).reindex(ob["timestamp"]).fillna(0.0)
    raw = imbalance.to_numpy(dtype=float) + 0.2 * signed_flow.to_numpy(dtype=float)
    s = pd.Series(raw, index=ob.index)
    m = s.rolling(window=window, min_periods=max(5, window // 2)).mean()
    sd = s.rolling(window=window, min_periods=max(5, window // 2)).std(ddof=0)
    z = (s - m) / (sd + 1e-6)
    z = z.clip(-3, 3)
    return np.tanh(z).rename("alpha")


def _compute_metrics() -> dict[str, Any]:
    ob = _prepare_orderbook(_load_csv(ORDERBOOK_PATH))
    tr = _prepare_trades(_load_csv(TRADES_PATH))
    if ob.empty:
        return {
            "pnl": 0.0,
            "ic": 0.0,
            "accuracy": 0.0,
            "trades": 0,
            "inventory": 0.0,
            "fees": 0.0,
            "K": 1.0,
            "spread": 0.5,
            "fill_prob": 1.0,
            "price_series": [],
            "pnl_series": [],
        }

    flow = _nearest_align_trades(ob, tr)
    ts = ob["timestamp"].to_numpy(dtype=np.int64)
    mid = ob["mid"].to_numpy(dtype=float)
    n_ob = len(ob)
    pnl_series = np.full(n_ob, np.nan, dtype=float)
    inventory = 0.0
    cash = 0.0
    fees_total = 0.0
    trades_processed = 0
    j = 0

    if not tr.empty:
        tr_sorted = tr.sort_values("timestamp").reset_index(drop=True)
        tr_ts = tr_sorted["timestamp"].to_numpy(dtype=np.int64)
        tr_px = tr_sorted["price"].to_numpy(dtype=float)
        tr_q = tr_sorted["quantity"].to_numpy(dtype=float)
        tr_side = tr_sorted["side"].astype(str).str.lower().to_numpy()
        n_tr = len(tr_sorted)

        for i in range(n_ob):
            t_snap = int(ts[i])
            while j < n_tr and int(tr_ts[j]) <= t_snap:
                p = float(tr_px[j])
                q = float(tr_q[j])
                if not (math.isfinite(p) and math.isfinite(q) and q > 0):
                    j += 1
                    continue
                notional = p * q
                fee = notional * FEE_RATE
                fees_total += fee
                s = str(tr_side[j])
                if s == "buy":
                    inventory += q
                    cash -= notional
                else:
                    inventory -= q
                    cash += notional
                cash -= fee
                inventory = float(np.clip(inventory, -100.0, 100.0))
                trades_processed += 1
                j += 1
            m = float(mid[i])
            if math.isfinite(m):
                pnl_series[i] = cash + inventory * m

        trades_n = trades_processed
        fees = float(fees_total)
        inventory = float(inventory)
    else:
        trades_n = 0
        fees = 0.0
        inventory = 0.0
        for i in range(n_ob):
            m = float(mid[i])
            if math.isfinite(m):
                pnl_series[i] = cash + inventory * m

    alpha = _compute_alpha_proxy(ob, flow)
    fwd = pd.Series(mid).shift(-3) - pd.Series(mid)
    frame = pd.DataFrame({"alpha": alpha, "fwd": fwd}).dropna()
    if len(frame) >= 5 and frame["alpha"].std() > 1e-12 and frame["fwd"].std() > 1e-12:
        ic = float(frame["alpha"].corr(frame["fwd"]))
    else:
        ic = 0.0
    if len(frame) >= 1:
        accuracy = float(((frame["alpha"] > 0) == (frame["fwd"] > 0)).mean())
    else:
        accuracy = 0.0

    keep = slice(-MAX_POINTS, None)
    price_series = [
        {"t": int(ts_i), "v": float(v)}
        for ts_i, v in zip(ts[keep], mid[keep], strict=False)
        if math.isfinite(v)
    ]
    pnl_series_out = [
        {"t": int(ts_i), "v": float(v)}
        for ts_i, v in zip(ts[keep], pnl_series[keep], strict=False)
        if math.isfinite(v)
    ]

    finite_pnl = pnl_series[np.isfinite(pnl_series)]
    pnl = float(finite_pnl[-1]) if len(finite_pnl) else 0.0
    try:
        k_series = estimate_liquidity_k(ob, tr, window=500)
        k_now = float(k_series.iloc[-1]) if len(k_series) else 1.0
    except Exception:
        k_now = 1.0
    k_now = float(np.clip(k_now, 0.01, 100.0))
    sigma_now = float(pd.Series(mid).rolling(window=100, min_periods=2).std(ddof=0).iloc[-1]) if len(mid) else 0.0
    if not math.isfinite(sigma_now):
        sigma_now = 0.0
    spread_now = float(max(0.1 * (sigma_now**2) + 1.0 / k_now, 0.5))
    fill_prob_now = float(np.exp(-k_now * (0.5 * spread_now)))
    return {
        "pnl": pnl,
        "ic": float(ic),
        "accuracy": float(accuracy),
        "trades": trades_n,
        "inventory": float(inventory),
        "fees": float(fees),
        "K": k_now,
        "spread": spread_now,
        "fill_prob": fill_prob_now,
        "price_series": price_series,
        "pnl_series": pnl_series_out,
    }


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return _compute_metrics()


@app.get("/")
def root() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")
