"""
Avellaneda–Stoikov market making engine.

Computes reservation price, optimal spread, and controlled bid/ask quotes
using only numpy and pandas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd

ArrayLike = Union[float, np.ndarray, pd.Series]


def _to_float_array(x: ArrayLike) -> np.ndarray:
    """Broadcast scalars to ndarray for uniform math."""
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=float, copy=False)
    return np.asarray(x, dtype=float)


@dataclass
class ASConfig:
    """Risk and microstructure parameters for Avellaneda–Stoikov."""

    gamma: float = 0.1
    """Risk aversion (inventory penalty). Must be > 0."""

    kappa: float = 5.0
    """Order arrival / liquidity parameter k in ln(1 + γ/k). Must be > 0."""

    sigma: float = 0.0
    """Per-step volatility of the mid-price (same units as mid). Must be >= 0."""

    time_horizon: float = 1.0
    """Remaining horizon τ = T − t (in the same time units as σ). Must be > 0."""

    # Compatibility fields (kept to avoid breaking current run.py wiring).
    spread_scale: float = 1.0
    gamma_inventory: float = 0.1

    # Engineering AS controls requested by strategy spec.
    sigma_window: int = 100
    min_total_spread: float = 0.5
    inventory_clip: float = 10.0
    min_k: float = 0.01
    max_k: float = 100.0

    def __post_init__(self) -> None:
        if self.gamma <= 0:
            raise ValueError("gamma must be positive")
        if self.kappa <= 0:
            raise ValueError("kappa must be positive")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")
        if self.time_horizon <= 0:
            raise ValueError("time_horizon must be positive")
        if self.sigma_window <= 1:
            raise ValueError("sigma_window must be > 1")
        if self.min_total_spread < 0:
            raise ValueError("min_total_spread must be >= 0")
        if self.inventory_clip <= 0:
            raise ValueError("inventory_clip must be > 0")
        if self.min_k <= 0 or self.max_k <= self.min_k:
            raise ValueError("invalid k clipping range")


class AvellanedaStoikovEngine:
    """
    Avellaneda–Stoikov optimal market making.

    Reservation price (skew away from mid when inventory is non-zero):

        r = s − q · γ · σ² · τ

    Total optimal spread (symmetric intensity k on bid and ask):

        δ = γ · σ² · τ + (2/γ) · ln(1 + γ/k)

    Quotes (symmetric half-spread):

        bid = r − δ/2,   ask = r + δ/2

    Inventory is measured in base asset units: positive = long, negative = short.
    """

    def __init__(self, config: ASConfig) -> None:
        self._config = config
        self._mid_hist: list[float] = []
        self._last_sigma: float = float(max(config.sigma, 0.0))
        self._last_quote: dict[str, float] = {}

    @property
    def config(self) -> ASConfig:
        return self._config

    def update_horizon(self, time_horizon: float) -> None:
        """Set remaining horizon τ (e.g. roll forward each step)."""
        if time_horizon <= 0:
            raise ValueError("time_horizon must be positive")
        self._config.time_horizon = time_horizon

    @property
    def last_quote(self) -> dict[str, float]:
        """Last point-in-time quote payload for downstream logging/debug."""
        return dict(self._last_quote)

    def _rolling_sigma_scalar(self, mid_value: float) -> float:
        cfg = self._config
        self._mid_hist.append(float(mid_value))
        if len(self._mid_hist) > cfg.sigma_window:
            self._mid_hist = self._mid_hist[-cfg.sigma_window :]
        if len(self._mid_hist) < 2:
            return max(self._last_sigma, cfg.eps if hasattr(cfg, "eps") else 0.0)
        sigma = float(np.nanstd(np.asarray(self._mid_hist, dtype=float), ddof=0))
        if not np.isfinite(sigma):
            sigma = self._last_sigma
        self._last_sigma = float(max(sigma, 0.0))
        return self._last_sigma

    def _rolling_sigma_series(self, mid: pd.Series) -> pd.Series:
        cfg = self._config
        s = mid.astype(float)
        sig = s.rolling(window=cfg.sigma_window, min_periods=2).std(ddof=0)
        if np.isfinite(cfg.sigma) and cfg.sigma > 0:
            sig = sig.fillna(float(cfg.sigma))
        else:
            sig = sig.fillna(method="bfill").fillna(0.0)
        return sig.clip(lower=0.0)

    def reservation_price(
        self,
        mid: ArrayLike,
        inventory: ArrayLike,
        alpha: ArrayLike | None = None,
    ) -> Union[float, np.ndarray, pd.Series]:
        """
        Reservation price r = s − q · γ · σ² · τ.

        Parameters
        ----------
        mid : float, ndarray, or Series
            Mid-price s.
        inventory : float, ndarray, or Series
            Inventory q (aligned with mid if array-like).
        """
        cfg = self._config
        s = _to_float_array(mid)
        q = np.clip(_to_float_array(inventory), -cfg.inventory_clip, cfg.inventory_clip)

        if isinstance(mid, pd.Series):
            sigma_arr = self._rolling_sigma_series(mid).to_numpy(dtype=float, copy=False)
        elif np.ndim(s) == 0:
            sigma_arr = np.asarray(self._rolling_sigma_scalar(float(s)), dtype=float)
        else:
            sig = pd.Series(s).rolling(window=cfg.sigma_window, min_periods=2).std(ddof=0)
            if np.isfinite(cfg.sigma) and cfg.sigma > 0:
                sig = sig.fillna(float(cfg.sigma))
            else:
                sig = sig.fillna(method="bfill").fillna(0.0)
            sigma_arr = sig.to_numpy(dtype=float, copy=False)

        # Engineering AS version:
        # r = mid - inventory * gamma * sigma^2
        r = s - q * cfg.gamma * (sigma_arr**2)
        if isinstance(mid, pd.Series):
            return pd.Series(r, index=mid.index, name=getattr(mid, "name", None))
        if np.ndim(r) == 0:
            return float(r)
        return r

    def _resolve_k(self, liquidity_k: ArrayLike | None = None) -> Union[float, np.ndarray]:
        cfg = self._config
        if liquidity_k is None:
            k = float(cfg.kappa)
            return float(np.clip(k, cfg.min_k, cfg.max_k))
        arr = _to_float_array(liquidity_k)
        arr = np.where(np.isfinite(arr), arr, cfg.kappa)
        arr = np.clip(arr, cfg.min_k, cfg.max_k)
        if np.ndim(arr) == 0:
            return float(arr)
        return arr

    def optimal_spread(
        self,
        mid: ArrayLike | None = None,
        liquidity_k: ArrayLike | None = None,
    ) -> Union[float, np.ndarray, pd.Series]:
        """
        Engineering AS spread:
            spread = max(gamma * sigma^2, min_total_spread)
        where sigma is rolling std(mid, window=sigma_window).
        """
        cfg = self._config
        k_eff = self._resolve_k(liquidity_k)
        if mid is None:
            spread = cfg.gamma * (max(cfg.sigma, 0.0) ** 2) + 1.0 / float(k_eff)
            return float(max(spread, cfg.min_total_spread))

        s = _to_float_array(mid)
        if isinstance(mid, pd.Series):
            sigma_arr = self._rolling_sigma_series(mid).to_numpy(dtype=float, copy=False)
            k_arr = np.asarray(k_eff, dtype=float)
            if np.ndim(k_arr) == 0:
                k_arr = np.full_like(sigma_arr, float(k_arr), dtype=float)
            spr = np.maximum(cfg.gamma * (sigma_arr**2) + 1.0 / k_arr, cfg.min_total_spread)
            return pd.Series(spr, index=mid.index, name="spread")
        if np.ndim(s) == 0:
            sigma = self._rolling_sigma_scalar(float(s))
            return float(max(cfg.gamma * (sigma**2) + 1.0 / float(k_eff), cfg.min_total_spread))

        sig = pd.Series(s).rolling(window=cfg.sigma_window, min_periods=2).std(ddof=0)
        if np.isfinite(cfg.sigma) and cfg.sigma > 0:
            sig = sig.fillna(float(cfg.sigma))
        else:
            sig = sig.fillna(method="bfill").fillna(0.0)
        sigma_arr = sig.to_numpy(dtype=float, copy=False)
        k_arr = np.asarray(k_eff, dtype=float)
        if np.ndim(k_arr) == 0:
            k_arr = np.full_like(sigma_arr, float(k_arr), dtype=float)
        spr = np.maximum(cfg.gamma * (sigma_arr**2) + 1.0 / k_arr, cfg.min_total_spread)
        return spr

    def half_spread(
        self,
        mid: ArrayLike | None = None,
        liquidity_k: ArrayLike | None = None,
    ) -> Union[float, np.ndarray, pd.Series]:
        """Half-spread δ/2 around the reservation price."""
        spr = self.optimal_spread(mid=mid, liquidity_k=liquidity_k)
        if isinstance(spr, pd.Series):
            return (0.5 * spr).rename("half_spread")
        arr = np.asarray(spr, dtype=float)
        if np.ndim(arr) == 0:
            return float(0.5 * arr)
        return 0.5 * arr

    def quote_prices(
        self,
        mid: ArrayLike,
        inventory: ArrayLike,
        alpha: ArrayLike | None = None,
        liquidity_k: ArrayLike | None = None,
    ) -> tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Symmetric bid/ask around reservation price.

        Returns
        -------
        bid, ask
        """
        cfg = self._config
        s = _to_float_array(mid)
        q = np.clip(_to_float_array(inventory), -cfg.inventory_clip, cfg.inventory_clip)

        _ = alpha  # compatibility: alpha hook kept, not used in engineering AS mode
        r = self.reservation_price(mid, q)
        h = self.half_spread(mid, liquidity_k=liquidity_k)
        if isinstance(r, pd.Series):
            bid = r - h
            ask = r + h
            return bid, ask
        rb = np.asarray(r, dtype=float) - h
        ra = np.asarray(r, dtype=float) + h
        if np.ndim(rb) == 0:
            return float(rb), float(ra)
        return rb, ra

    def quote_prices_clamped(
        self,
        mid: ArrayLike,
        inventory: ArrayLike,
        alpha: ArrayLike | None = None,
        liquidity_k: ArrayLike | None = None,
        min_half_spread: float = 0.0,
        max_inventory: float | None = None,
    ) -> tuple[Union[float, np.ndarray, pd.Series], Union[float, np.ndarray, pd.Series]]:
        """
        Same as quote_prices with optional inventory control:

        - Enforce a minimum half-spread (e.g. tick / fee floor).
        - If ``max_inventory`` is set, scale down skew contribution when |q| exceeds
          the cap by replacing q with sign(q) * min(|q|, max_inventory) in the
          reservation price (soft inventory cap).
        """
        cfg = self._config
        s = mid
        q = inventory
        if max_inventory is not None and max_inventory > 0:
            q_arr = _to_float_array(inventory)
            cap = min(float(max_inventory), cfg.inventory_clip)
            capped = np.sign(q_arr) * np.minimum(np.abs(q_arr), cap)
            if isinstance(inventory, pd.Series):
                q = pd.Series(capped, index=inventory.index)
            else:
                q = capped
        else:
            q_arr = _to_float_array(inventory)
            capped = np.sign(q_arr) * np.minimum(np.abs(q_arr), cfg.inventory_clip)
            if isinstance(inventory, pd.Series):
                q = pd.Series(capped, index=inventory.index)
            else:
                q = capped

        _ = alpha  # compatibility: alpha hook kept, not used in engineering AS mode
        r = self.reservation_price(s, q)
        # min_half_spread from backtest is preserved, but strategy floor on total spread
        # is always enforced via min_total_spread.
        h_floor = max(float(min_half_spread), 0.5 * cfg.min_total_spread)
        h = self.half_spread(s, liquidity_k=liquidity_k)
        if isinstance(r, pd.Series):
            hh = np.maximum(h.to_numpy(dtype=float, copy=False), h_floor)
            bid = r.to_numpy(dtype=float, copy=False) - hh
            ask = r.to_numpy(dtype=float, copy=False) + hh
            return pd.Series(bid, index=r.index), pd.Series(ask, index=r.index)
        hh = np.asarray(h, dtype=float)
        if np.ndim(hh) == 0:
            hh = float(max(float(hh), h_floor))
        else:
            hh = np.maximum(hh, h_floor)
        rb = np.asarray(r, dtype=float) - hh
        ra = np.asarray(r, dtype=float) + hh
        if np.ndim(rb) == 0:
            spread_val = float(2.0 * hh)
            k_cur = float(self._resolve_k(liquidity_k))
            delta = max(spread_val * 0.5, 0.0)
            fill_prob = float(np.exp(-k_cur * delta))
            self._last_quote = {
                "bid": float(rb),
                "ask": float(ra),
                "mid": float(_to_float_array(mid)),
                "inventory": float(np.asarray(q, dtype=float)),
                "K": k_cur,
                "spread": spread_val,
                "fill_prob": fill_prob,
            }
        if np.ndim(rb) == 0:
            return float(rb), float(ra)
        return rb, ra


def reservation_price(
    mid: ArrayLike,
    inventory: ArrayLike,
    gamma: float,
    sigma: float,
    time_horizon: float,
) -> Union[float, np.ndarray, pd.Series]:
    """Standalone reservation price: r = s − q · γ · σ² · τ."""
    s = _to_float_array(mid)
    q = _to_float_array(inventory)
    r = s - q * gamma * (sigma**2) * time_horizon
    if isinstance(mid, pd.Series):
        return pd.Series(r, index=mid.index, name=getattr(mid, "name", None))
    if np.ndim(r) == 0:
        return float(r)
    return r


def optimal_spread(
    gamma: float,
    kappa: float,
    sigma: float,
    time_horizon: float,
) -> float:
    """Standalone optimal total spread δ."""
    return AvellanedaStoikovEngine(
        ASConfig(gamma=gamma, kappa=kappa, sigma=sigma, time_horizon=time_horizon)
    ).optimal_spread()
