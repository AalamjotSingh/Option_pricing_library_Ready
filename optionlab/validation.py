"""Validation checks that return measured numbers rather than bare pass/fail.

Used by tests/test_pricing.py (assertions) and scripts/run_validation.py (report).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from . import binomial as bt
from . import black_scholes as bs
from . import greeks as gk
from . import monte_carlo as mc

BASE = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20)


def _random_params(rng, n):
    for _ in range(n):
        yield dict(S=100.0, K=float(rng.uniform(70, 130)), T=float(rng.uniform(0.1, 2.0)),
                   r=float(rng.uniform(0.0, 0.08)), sigma=float(rng.uniform(0.10, 0.60)),
                   q=float(rng.uniform(0.0, 0.04)))


def quad_price(S, K, T, r, sigma, q=0.0, option="call"):
    """Discounted risk-neutral expectation by numerical integration.

    Shares nothing with the closed-form derivation, so agreement is a real check.
    """
    mu = np.log(S) + (r - q - 0.5 * sigma**2) * T
    s = sigma * np.sqrt(T)
    sign = 1.0 if option == "call" else -1.0
    lo, hi = mu - 12 * s, mu + 12 * s
    pts = [np.log(K)] if lo < np.log(K) < hi else None
    val, _ = quad(lambda x: max(sign * (np.exp(x) - K), 0.0) * norm.pdf(x, mu, s),
                  lo, hi, points=pts, epsabs=1e-13, epsrel=1e-13, limit=200)
    return float(np.exp(-r * T) * val)


def check_black_scholes(seed=0, n=60):
    rng = np.random.default_rng(seed)
    quad_gap, parity_gap = [], []
    for p in _random_params(rng, n):
        c, put = bs.price(**p, option="call"), bs.price(**p, option="put")
        quad_gap += [abs(c - quad_price(**p, option="call")),
                     abs(put - quad_price(**p, option="put"))]
        parity = p["S"] * np.exp(-p["q"] * p["T"]) - p["K"] * np.exp(-p["r"] * p["T"])
        parity_gap.append(abs((c - put) - parity))
    return {
        "textbook_call": bs.price(42, 40, 0.5, 0.10, 0.20),
        "textbook_put": bs.price(42, 40, 0.5, 0.10, 0.20, option="put"),
        "max_abs_gap_vs_quadrature": max(quad_gap),
        "max_parity_gap": max(parity_gap),
        "n_parameter_sets": n,
    }


def check_tree(seed=0):
    rng = np.random.default_rng(seed)
    exact = bs.price(**BASE)
    steps = [10, 20, 40, 80, 160, 320, 640, 1280, 2560]
    raw, smooth = [], []
    for n in steps:
        a, b = bt.crr_price(**BASE, steps=n), bt.crr_price(**BASE, steps=n + 1)
        raw.append(abs(a - exact))
        smooth.append(abs(0.5 * (a + b) - exact))  # averaging N and N+1 removes odd/even oscillation
    slope_raw = float(np.polyfit(np.log(steps), np.log(raw), 1)[0])
    slope_smooth = float(np.polyfit(np.log(steps), np.log(smooth), 1)[0])

    parity_gap = 0.0
    for p in _random_params(rng, 10):
        c = bt.crr_price(**p, option="call", steps=300)
        put = bt.crr_price(**p, option="put", steps=300)
        ref = p["S"] * np.exp(-p["q"] * p["T"]) - p["K"] * np.exp(-p["r"] * p["T"])
        parity_gap = max(parity_gap, abs(c - put - ref))

    # Textbook two-step example (Hull): S=50, K=52, 20% up/down, r=5%, T=2y.
    hull_eu = bt.lattice_price(50, 52, 2, 0.05, 1.2, 0.8, 2, option="put")
    hull_am = bt.lattice_price(50, 52, 2, 0.05, 1.2, 0.8, 2, option="put", american=True)

    am_call = bt.crr_price(**BASE, option="call", american=True, steps=500)
    eu_call = bt.crr_price(**BASE, option="call", steps=500)
    div = dict(S=100.0, K=80.0, T=1.0, r=0.02, sigma=0.30, q=0.08)
    div_am = bt.crr_price(**div, option="call", american=True, steps=500)
    div_eu = bt.crr_price(**div, option="call", steps=500)
    am_put = {n: bt.crr_price(**BASE, option="put", american=True, steps=n)
              for n in (250, 500, 1000, 2000, 4000)}
    eu_put = bt.crr_price(**BASE, option="put", steps=4000)
    return {
        "steps": steps, "abs_error": raw, "abs_error_smoothed": smooth,
        "convergence_slope_raw": slope_raw, "convergence_slope_smoothed": slope_smooth,
        "max_parity_gap": parity_gap,
        "hull_european_put": hull_eu, "hull_american_put": hull_am,
        "american_call_minus_european_call_no_div": am_call - eu_call,
        "american_call_premium_with_div": div_am - div_eu,
        "american_put_by_steps": am_put,
        "american_put_early_exercise_premium": am_put[4000] - eu_put,
    }


def check_monte_carlo(seed=0, reps=500, n_paths=20_000):
    exact = bs.price(**BASE)
    rng = np.random.default_rng(seed)
    variants = {
        "plain": dict(antithetic=False, control_variate=False),
        "antithetic": dict(antithetic=True, control_variate=False),
        "control variate": dict(antithetic=False, control_variate=True),
        "antithetic + control variate": dict(antithetic=True, control_variate=True),
    }
    out = {}
    for name, kw in variants.items():
        est = np.empty(reps)
        covered = 0
        for i in range(reps):
            res = mc.mc_european(**BASE, option="call", n_paths=n_paths, rng=rng, **kw)
            est[i] = res.price
            covered += res.covers(exact)
        out[name] = {"coverage_95": covered / reps, "bias": float(est.mean() - exact),
                     "rmse": float(np.sqrt(np.mean((est - exact) ** 2))),
                     "variance": float(est.var(ddof=1))}
    for name in out:
        out[name]["variance_reduction_vs_plain"] = out["plain"]["variance"] / out[name]["variance"]
    return {"exact": exact, "reps": reps, "n_paths": n_paths, "variants": out}


def check_greeks(seed=0, n=40):
    rng = np.random.default_rng(seed)
    names = ["delta", "gamma", "vega", "theta", "rho"]
    worst = {k: 0.0 for k in names}
    for p in _random_params(rng, n):
        for opt in ("call", "put"):
            an = bs.greeks(**p, option=opt)
            fd = gk.fd_greeks(bs.price, option=opt, **p)
            for k in names:
                worst[k] = max(worst[k], abs(fd[k] - an[k]) / max(abs(an[k]), 1e-2))

    an = bs.greeks(**BASE)
    bumps = np.logspace(-8, -1, 15)
    sweep = [abs(gk.fd_greeks(bs.price, **BASE, option="call", rel_bump_S=h)["gamma"]
                 - an["gamma"]) / an["gamma"] for h in bumps]

    tree = {}
    for n_steps in (250, 1000, 4000):
        _, g = bt.crr_price(**BASE, steps=n_steps, return_greeks=True)
        tree[n_steps] = {k: abs(g[k] - an[k]) / abs(an[k]) for k in ("delta", "gamma", "theta")}

    def mc_price(**kw):
        return mc.mc_european(**kw, n_paths=400_000, seed=123).price

    f = gk.fd_greeks(mc_price, **BASE, option="call", rel_bump_S=0.01, bump_sigma=0.01)
    mc_err = {k: abs(f[k] - an[k]) / abs(an[k]) for k in ("delta", "vega")}
    return {"fd_vs_analytic_worst_scaled_error": worst, "gamma_bump_sizes": bumps.tolist(),
            "gamma_bump_errors": sweep, "tree_rel_error": tree,
            "mc_common_random_numbers_rel_error": mc_err}


def run_all(seed=0):
    return {"black_scholes": check_black_scholes(seed), "tree": check_tree(seed),
            "monte_carlo": check_monte_carlo(seed), "greeks": check_greeks(seed)}
