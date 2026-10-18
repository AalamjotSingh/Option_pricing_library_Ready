"""Recombining binomial trees (European and American) with tree-based Greeks.

`lattice_price` takes explicit up/down factors (so textbook examples can be
reproduced exactly); `crr_price` uses Cox-Ross-Rubinstein factors
u = exp(sigma*sqrt(dt)), d = 1/u. Backward induction is vectorised over each
time slice, so a 5,000-step tree takes well under a second.
"""
from __future__ import annotations

import numpy as np


def lattice_price(S, K, T, r, u, d, steps, q=0.0, option="call", american=False,
                  return_greeks=False):
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'")
    if steps < (2 if return_greeks else 1):
        raise ValueError("not enough steps")
    is_call = option == "call"
    dt = T / steps
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError(f"risk-neutral probability {p:.4f} is outside (0, 1); use more steps")
    disc = np.exp(-r * dt)

    def payoff(x):
        return np.maximum(x - K, 0.0) if is_call else np.maximum(K - x, 0.0)

    j = np.arange(steps + 1)
    nodes = S * u**j * d ** (steps - j)          # terminal stock prices, low to high
    V = payoff(nodes)
    snaps = {}
    if return_greeks and steps == 2:
        snaps[2] = (V.copy(), nodes.copy())
    for i in range(steps - 1, -1, -1):
        V = disc * (p * V[1:] + (1.0 - p) * V[:-1])
        nodes = nodes[1:] / u                      # stock prices at step i
        if american:
            V = np.maximum(V, payoff(nodes))
        if return_greeks and i in (1, 2):
            snaps[i] = (V.copy(), nodes.copy())
    value = float(V[0])
    if not return_greeks:
        return value

    (V1, n1), (V2, n2) = snaps[1], snaps[2]
    delta = (V1[1] - V1[0]) / (n1[1] - n1[0])
    half_span = 0.5 * (n2[2] - n2[0])
    gamma = ((V2[2] - V2[1]) / (n2[2] - n2[1]) - (V2[1] - V2[0]) / (n2[1] - n2[0])) / half_span
    # With u*d == 1 the middle node at step 2 has the original spot, so the
    # value change over two steps gives theta (per year) without extra revaluation.
    theta = (V2[1] - value) / (2.0 * dt) if abs(u * d - 1.0) < 1e-12 else float("nan")
    return value, {"delta": float(delta), "gamma": float(gamma), "theta": float(theta)}


def crr_price(S, K, T, r, sigma, q=0.0, option="call", american=False, steps=500,
              return_greeks=False):
    dt = T / steps
    u = float(np.exp(sigma * np.sqrt(dt)))
    return lattice_price(S, K, T, r, u, 1.0 / u, steps, q=q, option=option,
                         american=american, return_greeks=return_greeks)
