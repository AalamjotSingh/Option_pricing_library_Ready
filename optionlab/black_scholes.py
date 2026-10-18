"""Closed-form Black-Scholes-Merton prices and Greeks for European options.

Conventions: S spot, K strike, T years to expiry, r continuously compounded
risk-free rate, sigma annualised volatility, q continuous dividend yield.
Greeks are per unit of the input: vega per 1.00 of sigma (divide by 100 for one
vol point), rho per 1.00 of r, theta per year in calendar time (the derivative
with respect to elapsed time, so it is negative for most long options).
Everything broadcasts over NumPy arrays and returns floats for scalar input.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _is_call(option: str) -> bool:
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'")
    return option == "call"


def _arrays(*xs):
    return tuple(np.asarray(x, dtype=float) for x in xs)


def _out(x):
    x = np.asarray(x)
    return float(x) if x.ndim == 0 else x


def d1_d2(S, K, T, r, sigma, q=0.0):
    S, K, T, r, sigma, q = _arrays(S, K, T, r, sigma, q)
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def price(S, K, T, r, sigma, q=0.0, option="call"):
    is_call = _is_call(option)
    S, K, T, r, sigma, q = _arrays(S, K, T, r, sigma, q)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    df_r, df_q = np.exp(-r * T), np.exp(-q * T)
    if is_call:
        val = S * df_q * ndtr(d1) - K * df_r * ndtr(d2)
    else:
        val = K * df_r * ndtr(-d2) - S * df_q * ndtr(-d1)
    return _out(val)


def delta(S, K, T, r, sigma, q=0.0, option="call"):
    is_call = _is_call(option)
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    df_q = np.exp(-np.asarray(q, dtype=float) * np.asarray(T, dtype=float))
    return _out(df_q * ndtr(d1) if is_call else -df_q * ndtr(-d1))


def greeks(S, K, T, r, sigma, q=0.0, option="call"):
    """Delta, gamma, vega, theta (per year) and rho as a dict."""
    is_call = _is_call(option)
    S, K, T, r, sigma, q = _arrays(S, K, T, r, sigma, q)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    sqrt_t = np.sqrt(T)
    df_r, df_q = np.exp(-r * T), np.exp(-q * T)
    pdf_d1 = _INV_SQRT_2PI * np.exp(-0.5 * d1 * d1)
    gamma = df_q * pdf_d1 / (S * sigma * sqrt_t)
    vega = S * df_q * pdf_d1 * sqrt_t
    decay = -S * df_q * pdf_d1 * sigma / (2.0 * sqrt_t)
    if is_call:
        dlt = df_q * ndtr(d1)
        theta = decay - r * K * df_r * ndtr(d2) + q * S * df_q * ndtr(d1)
        rho = K * T * df_r * ndtr(d2)
    else:
        dlt = -df_q * ndtr(-d1)
        theta = decay + r * K * df_r * ndtr(-d2) - q * S * df_q * ndtr(-d1)
        rho = -K * T * df_r * ndtr(-d2)
    return {"delta": _out(dlt), "gamma": _out(gamma), "vega": _out(vega),
            "theta": _out(theta), "rho": _out(rho)}
