"""Monte Carlo pricing of European options under GBM with variance reduction."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MCResult:
    price: float
    stderr: float
    n_paths: int

    @property
    def ci95(self):
        return (self.price - 1.96 * self.stderr, self.price + 1.96 * self.stderr)

    def covers(self, value: float) -> bool:
        lo, hi = self.ci95
        return lo <= value <= hi


def mc_european(S, K, T, r, sigma, q=0.0, option="call", n_paths=100_000,
                antithetic=True, control_variate=False, rng=None, seed=None):
    """Price a European option from terminal GBM draws.

    antithetic: pair each normal draw z with -z and average the pair, so each
        sample is one pair (n_paths is the number of payoffs evaluated).
    control_variate: use the discounted terminal stock price, whose mean
        S*exp(-q*T) is known exactly, as a control for the discounted payoff.
    Passing the same `seed` reuses the same random numbers (common random
    numbers), which is what makes bump-and-revalue Greeks usable.
    """
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'")
    is_call = option == "call"
    rng = rng if rng is not None else np.random.default_rng(seed)
    n = n_paths // 2 if antithetic else n_paths
    z = rng.standard_normal(n)
    drift = (r - q - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)
    disc = np.exp(-r * T)

    def payoff(x):
        return np.maximum(x - K, 0.0) if is_call else np.maximum(K - x, 0.0)

    s_up = S * np.exp(drift + vol * z)
    if antithetic:
        s_dn = S * np.exp(drift - vol * z)
        y = 0.5 * (payoff(s_up) + payoff(s_dn)) * disc
        x = 0.5 * (s_up + s_dn) * disc
    else:
        y = payoff(s_up) * disc
        x = s_up * disc
    if control_variate:
        c = np.cov(y, x, ddof=1)
        beta = c[0, 1] / c[1, 1]
        y = y - beta * (x - S * np.exp(-q * T))
    return MCResult(float(y.mean()), float(y.std(ddof=1) / np.sqrt(n)),
                    n * (2 if antithetic else 1))
