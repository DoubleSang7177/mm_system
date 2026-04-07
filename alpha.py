"""
强化版多档 OFI alpha：加权、滚动统计、z-score、去噪、tanh → [-1,1]。
兼容 run.py：XGBoostAlphaConfig / XGBoostAlphaModel 接口（无 XGBoost 训练）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


def _norm_lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _level_volume_array(df: pd.DataFrame, side: str, n_levels: int) -> np.ndarray:
    """(n_rows, n_levels) 向量化；缺列用 0。"""
    n = len(df)
    out = np.zeros((n, n_levels), dtype=np.float64)
    for i in range(1, n_levels + 1):
        candidates = (
            f"{side}_volume_{i}",
            f"{side}_size_{i}",
            f"{side}_vol_{i}",
        )
        col = None
        for c in candidates:
            if c in df.columns:
                col = c
                break
        if col is None and i == 1:
            fallback = "bid_size" if side == "bid" else "ask_size"
            if fallback in df.columns:
                col = fallback
        if col is not None:
            out[:, i - 1] = df[col].to_numpy(dtype=np.float64, copy=False)
    return out


def _level_weights(n_levels: int, mode: str, decay_lambda: float) -> np.ndarray:
    if mode == "exp":
        w = np.exp(-decay_lambda * np.arange(n_levels, dtype=np.float64))
    else:
        w = 1.0 / np.arange(1, n_levels + 1, dtype=np.float64)
    return w


def _rolling_mean_np(x: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(x)
    return s.rolling(window=window, min_periods=max(2, window // 2)).mean().to_numpy()


def _rolling_std_np(x: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(x)
    return s.rolling(window=window, min_periods=max(2, window // 2)).std(ddof=0).to_numpy()


def compute_weighted_ofi_per_level_vec(df: pd.DataFrame, n_levels: int, noise_threshold: float, weights: np.ndarray):
    """纯向量化 Δ（与 df 行对齐）。"""
    bv = _level_volume_array(df, "bid", n_levels)
    av = _level_volume_array(df, "ask", n_levels)
    db = np.zeros_like(bv)
    da = np.zeros_like(av)
    db[1:] = np.diff(bv, axis=0)
    da[1:] = np.diff(av, axis=0)
    small = (np.abs(db) + np.abs(da)) < noise_threshold
    db = np.where(small, 0.0, db)
    da = np.where(small, 0.0, da)
    ofi_l = db - da
    ofi_total = (ofi_l * weights.reshape(1, -1)).sum(axis=1)
    return ofi_l, ofi_total


@dataclass
class OFIAlphaConfig:
    n_levels: int = 5
    window: int = 10
    noise_threshold: float = 1e-9
    clip_z: float = 3.0
    weight_mode: str = "inverse_level"
    decay_lambda: float = 0.35
    eps: float = 1e-6
    horizon: int = 3
    train_fraction: float = 0.7
    vol_window: int = 20
    lookback: int = 10
    n_estimators: int = 400
    max_depth: int = 4
    learning_rate: float = 0.04
    subsample: float = 0.75
    colsample_bytree: float = 0.75
    min_child_weight: float = 5.0
    reg_lambda: float = 2.5
    reg_alpha: float = 0.4
    random_state: int = 42
    n_jobs: int = 0
    eval_metric: str = "logloss"
    early_stopping_rounds: int = 60
    momentum_lags: tuple[int, ...] = (1, 3, 5)
    extra_params: dict[str, Any] = field(default_factory=dict)


def compute_alpha(df: pd.DataFrame, config: OFIAlphaConfig | None = None) -> pd.Series:
    """
    多档加权 OFI → 滚动 z-score → clip → tanh，输出 alpha ∈ [-1, 1]。
    """
    cfg = config or OFIAlphaConfig()
    d = _norm_lower_cols(df)
    n_levels = int(cfg.n_levels)
    w = _level_weights(n_levels, cfg.weight_mode, cfg.decay_lambda)
    _, ofi_total = compute_weighted_ofi_per_level_vec(
        d, n_levels, cfg.noise_threshold, w
    )
    idx = d.index
    mu = _rolling_mean_np(ofi_total, cfg.window)
    sd = _rolling_std_np(ofi_total, cfg.window)
    sd = np.where(sd < cfg.eps, cfg.eps, sd)
    ofi_z = (ofi_total - mu) / sd
    ofi_z = np.clip(ofi_z, -cfg.clip_z, cfg.clip_z)
    alpha = np.tanh(ofi_z)
    out = pd.Series(alpha, index=idx, name="alpha")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def forward_log_return(mid: pd.Series, horizon: int = 3) -> pd.Series:
    m = mid.astype(float)
    return (np.log(m.shift(-horizon)) - np.log(m)).rename(f"fwd_log_ret_{horizon}")


def forward_price_up_label(mid: pd.Series, horizon: int = 3) -> pd.Series:
    m = mid.astype(float)
    future = m.shift(-horizon)
    return (future > m).astype(np.int32).rename("y_up")


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def evaluate_alpha_signal(
    alpha: pd.Series,
    mid: pd.Series,
    horizon: int = 3,
    n_buckets: int = 3,
) -> dict[str, Any]:
    fr = forward_log_return(mid, horizon=horizon)
    frame = pd.DataFrame({"alpha": alpha.astype(float), "fwd": fr})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < max(30, horizon + 5):
        return {
            "ic_pearson": float("nan"),
            "ic_spearman": float("nan"),
            "accuracy_direction": float("nan"),
            "n": float(len(frame)),
            "bucket_mean_return": {},
        }

    a = frame["alpha"].to_numpy()
    f = frame["fwd"].to_numpy()
    ic_p = _safe_corr(a, f)
    ranks_a = pd.Series(a).rank().to_numpy()
    ranks_f = pd.Series(f).rank().to_numpy()
    ic_s = _safe_corr(ranks_a, ranks_f)
    pred_up = a > 0.0
    act_up = f > 0.0
    acc = float(np.mean(pred_up == act_up))
    bucket_means: dict[str, float] = {}
    try:
        cats = pd.qcut(frame["alpha"], q=n_buckets, labels=False, duplicates="drop")
        for k in sorted(pd.Series(cats).dropna().unique()):
            msk = cats == k
            bucket_means[f"bucket_{int(k)}"] = float(np.mean(f[msk.to_numpy()]))
    except (ValueError, TypeError):
        pass

    return {
        "ic_pearson": ic_p,
        "ic_spearman": ic_s,
        "accuracy_direction": acc,
        "n": float(len(frame)),
        "bucket_mean_return": bucket_means,
    }


def build_ml_feature_matrix(
    book: pd.DataFrame,
    vol_window: int = 20,
    lookback: int = 10,
    momentum_lags: tuple[int, ...] = (1, 3, 5),
    eps: float = 1e-12,
) -> pd.DataFrame:
    """研究用特征表：强化 OFI 分量 + 滚动量（与 vol_window/lookback 对齐）。"""
    cfg = OFIAlphaConfig(window=lookback, vol_window=vol_window)
    d = _norm_lower_cols(book)
    n_levels = cfg.n_levels
    w = _level_weights(n_levels, cfg.weight_mode, cfg.decay_lambda)
    ofi_l, ofi_total = compute_weighted_ofi_per_level_vec(
        d, n_levels, cfg.noise_threshold, w
    )
    roll_sum = pd.Series(ofi_total).rolling(lookback, min_periods=max(2, lookback // 2)).sum()
    roll_m = pd.Series(ofi_total).rolling(lookback, min_periods=max(2, lookback // 2)).mean()
    roll_s = pd.Series(ofi_total).rolling(lookback, min_periods=max(2, lookback // 2)).std(ddof=0)

    cols: dict[str, pd.Series] = {
        "ofi_total_raw": pd.Series(ofi_total, index=d.index),
        "ofi_roll_sum": roll_sum,
        "ofi_roll_mean": roll_m,
        "ofi_roll_std": roll_s,
        "alpha_ofi": compute_alpha(book, OFIAlphaConfig(window=lookback, n_levels=n_levels)),
    }
    for j in range(n_levels):
        cols[f"ofi_level_{j+1}"] = pd.Series(ofi_l[:, j], index=d.index)

    return pd.DataFrame(cols, index=d.index)


XGBoostAlphaConfig = OFIAlphaConfig
AlphaMLConfig = OFIAlphaConfig


class XGBoostAlphaModel:
    """无监督 OFI alpha：fit 仅保存配置；预测为 compute_alpha。"""

    def __init__(self, config: OFIAlphaConfig | None = None) -> None:
        self._config = config or OFIAlphaConfig()
        self._fitted = False

    @property
    def config(self) -> OFIAlphaConfig:
        return self._config

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit_time_split(
        self,
        book: pd.DataFrame,
        config: OFIAlphaConfig | None = None,
    ) -> tuple["XGBoostAlphaModel", dict[str, Any]]:
        cfg = config or self._config
        self._config = cfg
        d = _norm_lower_cols(book)
        if "bid_price" not in d.columns or "ask_price" not in d.columns:
            raise ValueError("need bid_price and ask_price for evaluation")
        mid = (d["bid_price"].astype(float) + d["ask_price"].astype(float)) * 0.5
        alpha_full = compute_alpha(book, cfg)
        valid = alpha_full.notna() & mid.notna()
        idx = alpha_full.index[valid.to_numpy()]
        n = int(len(idx))
        if n < 80:
            raise ValueError(f"need >=80 valid rows; got {n}")
        n_train = max(int(n * cfg.train_fraction), 30)
        n_train = min(n_train, n - 20)
        iloc_tr = idx[:n_train]
        iloc_te = idx[n_train:]
        alpha_te = alpha_full.reindex(iloc_te)
        mid_te = mid.reindex(iloc_te)
        ev = evaluate_alpha_signal(alpha_te, mid_te, horizon=cfg.horizon, n_buckets=3)
        y_up = forward_price_up_label(mid, horizon=cfg.horizon).reindex(iloc_te)
        proba_like = (alpha_te + 1.0) / 2.0
        msk = proba_like.notna() & y_up.notna()
        if msk.sum() > 0:
            ev["classification_accuracy"] = float(
                np.mean((proba_like[msk].to_numpy() > 0.5) == (y_up[msk].to_numpy() == 1))
            )
        else:
            ev["classification_accuracy"] = float("nan")
        ev["train_rows"] = float(n_train)
        ev["test_rows"] = float(len(iloc_te))
        self._fitted = True
        return self, ev

    def fit_full(
        self,
        book: pd.DataFrame,
        config: OFIAlphaConfig | None = None,
    ) -> "XGBoostAlphaModel":
        cfg = config or self._config
        self._config = cfg
        self._fitted = True
        return self

    def predict_proba_up(self, book: pd.DataFrame) -> pd.Series:
        a = compute_alpha(book, self._config)
        p = (a + 1.0) / 2.0
        return p.clip(0.0, 1.0).rename("alpha_proba_up")

    def predict_alpha_score(self, book: pd.DataFrame) -> pd.Series:
        return compute_alpha(book, self._config)


AlphaMLModel = XGBoostAlphaModel


def train_alpha_model(
    book: pd.DataFrame,
    config: OFIAlphaConfig | None = None,
) -> XGBoostAlphaModel:
    return XGBoostAlphaModel(config=config).fit_full(book)

