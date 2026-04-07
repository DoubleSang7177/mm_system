"""
Multi-level OFI + Trade Flow alpha (strictly causal).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from features import multi_level_ofi, rolling_zscore, trade_flow


def _norm_lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


@dataclass
class OFIAlphaConfig:
    # Multi-level OFI
    ofi_levels: tuple[int, int, int] = (1, 2, 3)
    ofi_weights: tuple[float, float, float] = (0.6, 0.3, 0.1)
    noise_threshold: float = 0.0

    # Trade flow + normalization
    zscore_window: int = 50
    clip_z: float = 4.0
    eps: float = 1e-6

    # alpha = a * OFI_total_z + b * TradeFlow_z
    a: float = 0.7
    b: float = 0.3
    auto_flip_sign: bool = True

    # evaluation and compatibility fields
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


def _safe_std(s: pd.Series) -> pd.Series:
    out = s.astype(float)
    return out.replace([np.inf, -np.inf], np.nan)


def _extract_trade_flow(df: pd.DataFrame) -> pd.Series:
    d = _norm_lower_cols(df)
    if "buy_volume" in d.columns and "sell_volume" in d.columns:
        return trade_flow(d["buy_volume"], d["sell_volume"])
    if "trade_flow" in d.columns:
        return pd.to_numeric(d["trade_flow"], errors="coerce").astype(float).fillna(0.0).rename("trade_flow")
    return pd.Series(0.0, index=d.index, dtype=float, name="trade_flow")


def build_ml_feature_matrix(
    book: pd.DataFrame,
    vol_window: int = 20,
    lookback: int = 10,
    momentum_lags: tuple[int, ...] = (1, 3, 5),
    eps: float = 1e-12,
) -> pd.DataFrame:
    cfg = OFIAlphaConfig(zscore_window=max(lookback, 20))
    d = _norm_lower_cols(book)

    ofi_df = multi_level_ofi(
        d,
        levels=cfg.ofi_levels,
        weights=cfg.ofi_weights,
        noise_threshold=cfg.noise_threshold,
    )
    tf = _extract_trade_flow(d)

    ofi_l1_z = rolling_zscore(ofi_df["ofi_l1"], window=cfg.zscore_window).rename("ofi_l1_z")
    ofi_l2_z = rolling_zscore(ofi_df["ofi_l2"], window=cfg.zscore_window).rename("ofi_l2_z")
    ofi_l3_z = rolling_zscore(ofi_df["ofi_l3"], window=cfg.zscore_window).rename("ofi_l3_z")
    ofi_total_z = rolling_zscore(ofi_df["ofi_total"], window=cfg.zscore_window).rename("ofi_total_z")
    trade_flow_z = rolling_zscore(tf, window=cfg.zscore_window).rename("trade_flow_z")

    feature_df = pd.DataFrame(
        {
            "ofi_l1": _safe_std(ofi_df["ofi_l1"]),
            "ofi_l2": _safe_std(ofi_df["ofi_l2"]),
            "ofi_l3": _safe_std(ofi_df["ofi_l3"]),
            "ofi_total": _safe_std(ofi_df["ofi_total"]),
            "trade_flow": _safe_std(tf),
            "ofi_l1_z": _safe_std(ofi_l1_z).clip(-cfg.clip_z, cfg.clip_z),
            "ofi_l2_z": _safe_std(ofi_l2_z).clip(-cfg.clip_z, cfg.clip_z),
            "ofi_l3_z": _safe_std(ofi_l3_z).clip(-cfg.clip_z, cfg.clip_z),
            "ofi_total_z": _safe_std(ofi_total_z).clip(-cfg.clip_z, cfg.clip_z),
            "trade_flow_z": _safe_std(trade_flow_z).clip(-cfg.clip_z, cfg.clip_z),
        },
        index=d.index,
    )
    return feature_df


def compute_alpha(df: pd.DataFrame, config: OFIAlphaConfig | None = None) -> pd.Series:
    cfg = config or OFIAlphaConfig()
    d = _norm_lower_cols(df)
    ofi_df = multi_level_ofi(
        d,
        levels=cfg.ofi_levels,
        weights=cfg.ofi_weights,
        noise_threshold=cfg.noise_threshold,
    )
    tf = _extract_trade_flow(d)

    ofi_total_z = rolling_zscore(ofi_df["ofi_total"], window=cfg.zscore_window)
    trade_flow_z = rolling_zscore(tf, window=cfg.zscore_window)
    ofi_total_z = (
        ofi_total_z.replace([np.inf, -np.inf], np.nan)
        .clip(-cfg.clip_z, cfg.clip_z)
        .fillna(0.0)
    )
    trade_flow_z = (
        trade_flow_z.replace([np.inf, -np.inf], np.nan)
        .clip(-cfg.clip_z, cfg.clip_z)
        .fillna(0.0)
    )

    alpha_raw = cfg.a * ofi_total_z + cfg.b * trade_flow_z
    alpha = np.tanh(alpha_raw)
    return pd.Series(alpha, index=d.index, name="alpha").clip(-1.0, 1.0)


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


XGBoostAlphaConfig = OFIAlphaConfig
AlphaMLConfig = OFIAlphaConfig


class XGBoostAlphaModel:
    def __init__(self, config: OFIAlphaConfig | None = None) -> None:
        self._config = config or OFIAlphaConfig()
        self._fitted = False
        self._alpha_sign = 1.0

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

        if cfg.auto_flip_sign:
            tr_eval = evaluate_alpha_signal(
                alpha_full.reindex(iloc_tr),
                mid.reindex(iloc_tr),
                horizon=cfg.horizon,
                n_buckets=3,
            )
            ic_tr = float(tr_eval.get("ic_pearson", np.nan))
            if np.isfinite(ic_tr) and ic_tr < 0.0:
                self._alpha_sign = -1.0
            else:
                self._alpha_sign = 1.0
        else:
            self._alpha_sign = 1.0

        alpha_te = self._alpha_sign * alpha_full.reindex(iloc_te)
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
        a = self._alpha_sign * compute_alpha(book, self._config)
        p = (a + 1.0) / 2.0
        return p.clip(0.0, 1.0).rename("alpha_proba_up")

    def predict_alpha_score(self, book: pd.DataFrame) -> pd.Series:
        return (self._alpha_sign * compute_alpha(book, self._config)).rename("alpha")


AlphaMLModel = XGBoostAlphaModel


def train_alpha_model(
    book: pd.DataFrame,
    config: OFIAlphaConfig | None = None,
) -> XGBoostAlphaModel:
    return XGBoostAlphaModel(config=config).fit_full(book)

