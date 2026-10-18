"""Bump-and-revalue (central finite difference) Greeks for any pricing function."""
from __future__ import annotations


def fd_greeks(price_fn, S, K, T, r, sigma, q=0.0, option="call",
              rel_bump_S=1e-3, bump_sigma=1e-4, bump_r=1e-4, bump_T=1e-4):
    """Central differences around the base point.

    `price_fn` must accept S, K, T, r, sigma, q, option as keyword arguments.
    Theta is the calendar-time derivative (minus the derivative in T) per year.
    Units match optionlab.black_scholes.greeks.
    """
    base = dict(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option=option)

    def f(**override):
        return price_fn(**{**base, **override})

    hS = rel_bump_S * S
    hT = min(bump_T, 0.5 * T)
    p0, up, dn = f(), f(S=S + hS), f(S=S - hS)
    return {
        "delta": (up - dn) / (2 * hS),
        "gamma": (up - 2 * p0 + dn) / hS**2,
        "vega": (f(sigma=sigma + bump_sigma) - f(sigma=sigma - bump_sigma)) / (2 * bump_sigma),
        "theta": -(f(T=T + hT) - f(T=T - hT)) / (2 * hT),
        "rho": (f(r=r + bump_r) - f(r=r - bump_r)) / (2 * bump_r),
    }
