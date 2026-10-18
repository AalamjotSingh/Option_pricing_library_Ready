"""Discrete delta-hedging simulation for a short European call.

The seller receives the Black-Scholes premium, holds delta shares, and
rebalances every `every` fine time steps. Each trade costs `cost_rate` times
the traded notional, cash earns the risk-free rate, and at expiry the stock
position is liquidated and the option payoff is paid. Hedging P&L is measured
at expiry and discounted to t=0 (no dividends).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import black_scholes as bs


def simulate_gbm(S0, mu, sigma, T, n_steps, n_paths, rng):
    """Exact GBM paths on a uniform grid; returns shape (n_paths, n_steps + 1)."""
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_s = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_ret, axis=1)], axis=1)
    return S0 * np.exp(log_s)


@dataclass
class HedgeResult:
    premium: float
    pnl: np.ndarray            # PV hedging P&L per path (hedge portfolio minus payoff)
    cost: np.ndarray           # PV of all transaction costs per path
    interior_cost: np.ndarray  # PV of costs from rebalances strictly inside (0, T)
    n_intervals: int


def delta_hedge_short_call(paths, K, T, r, sigma_hedge, every=1, cost_rate=0.0):
    n_paths, n_cols = paths.shape
    n_fine = n_cols - 1
    if n_fine % every:
        raise ValueError("`every` must divide the number of fine time steps")
    idx = np.arange(0, n_fine + 1, every)
    S = paths[:, idx]
    n_int = len(idx) - 1
    dt = T * every / n_fine
    t = np.arange(n_int + 1) * dt
    premium = bs.price(paths[0, 0], K, T, r, sigma_hedge)

    cash = np.full(n_paths, premium)
    shares = np.zeros(n_paths)
    cost = np.zeros(n_paths)
    interior = np.zeros(n_paths)
    growth = np.exp(r * dt)
    for k in range(n_int):
        if k > 0:
            cash *= growth
        target = bs.delta(S[:, k], K, T - t[k], r, sigma_hedge)
        trade = target - shares
        fee = cost_rate * np.abs(trade) * S[:, k]
        cash -= trade * S[:, k] + fee
        pv = np.exp(-r * t[k])
        cost += pv * fee
        if k > 0:
            interior += pv * fee
        shares = target
    cash *= growth
    S_T = S[:, -1]
    fee_T = cost_rate * np.abs(shares) * S_T
    cash += shares * S_T - fee_T
    cost += np.exp(-r * T) * fee_T
    pnl_T = cash - np.maximum(S_T - K, 0.0)
    return HedgeResult(premium, np.exp(-r * T) * pnl_T, cost, interior, n_int)


def summarize(res: HedgeResult) -> dict:
    """Headline statistics, all as a percentage of the option premium."""
    scale = 100.0 / res.premium
    return {
        "n_intervals": res.n_intervals,
        "mean_pnl_pct": float(res.pnl.mean() * scale),
        "std_pnl_pct": float(res.pnl.std(ddof=1) * scale),
        "p05_pnl_pct": float(np.quantile(res.pnl, 0.05) * scale),
        "mean_cost_pct": float(res.cost.mean() * scale),
        "mean_interior_cost_pct": float(res.interior_cost.mean() * scale),
    }
