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

    gamma: float
    """Risk aversion (inventory penalty). Must be > 0."""

    kappa: float
    """Order arrival / liquidity parameter k in ln(1 + γ/k). Must be > 0."""

    sigma: float
    """Per-step volatility of the mid-price (same units as mid). Must be >= 0."""

    time_horizon: float
    """Remaining horizon τ = T − t (in the same time units as σ). Must be > 0."""

    def __post_init__(self) -> None:
        if self.gamma <= 0:
            raise ValueError("gamma must be positive")
        if self.kappa <= 0:
            raise ValueError("kappa must be positive")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")
        if self.time_horizon <= 0:
            raise ValueError("time_horizon must be positive")


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

    @property
    def config(self) -> ASConfig:
        return self._config

    def update_horizon(self, time_horizon: float) -> None:
        """Set remaining horizon τ (e.g. roll forward each step)."""
        if time_horizon <= 0:
            raise ValueError("time_horizon must be positive")
        self._config.time_horizon = time_horizon

    def reservation_price(
        self,
        mid: ArrayLike,
        inventory: ArrayLike,
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
        q = _to_float_array(inventory)
        tau = cfg.time_horizon
        r = s - q * cfg.gamma * (cfg.sigma**2) * tau
        if isinstance(mid, pd.Series):
            return pd.Series(r, index=mid.index, name=getattr(mid, "name", None))
        if np.ndim(r) == 0:
            return float(r)
        return r

    def optimal_spread(self) -> float:
        """
        Total spread δ = γ · σ² · τ + (2/γ) · ln(1 + γ/k).

        Returns a scalar; τ, γ, k, σ come from config.
        """
        cfg = self._config
        tau = cfg.time_horizon
        intensity_term = (2.0 / cfg.gamma) * np.log(1.0 + cfg.gamma / cfg.kappa)
        return cfg.gamma * (cfg.sigma**2) * tau + intensity_term

    def half_spread(self) -> float:
        """Half-spread δ/2 around the reservation price."""
        return 0.5 * self.optimal_spread()

    def quote_prices(
        self,
        mid: ArrayLike,
        inventory: ArrayLike,
    ) -> tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Symmetric bid/ask around reservation price.

        Returns
        -------
        bid, ask
        """
        r = self.reservation_price(mid, inventory)
        h = self.half_spread()
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
            capped = np.sign(q_arr) * np.minimum(np.abs(q_arr), max_inventory)
            if isinstance(inventory, pd.Series):
                q = pd.Series(capped, index=inventory.index)
            else:
                q = capped

        r = self.reservation_price(s, q)
        h = max(self.half_spread(), min_half_spread)
        if isinstance(r, pd.Series):
            return r - h, r + h
        rb = np.asarray(r, dtype=float) - h
        ra = np.asarray(r, dtype=float) + h
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
