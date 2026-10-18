"""Assertions over optionlab.validation. Thresholds are stated next to each check."""
import numpy as np
import pytest

from optionlab import binomial as bt
from optionlab import black_scholes as bs
from optionlab import validation as v


@pytest.fixture(scope="module")
def bs_res():
    return v.check_black_scholes()


@pytest.fixture(scope="module")
def tree_res():
    return v.check_tree()


@pytest.fixture(scope="module")
def mc_res():
    return v.check_monte_carlo()


@pytest.fixture(scope="module")
def greek_res():
    return v.check_greeks()


# ---- Black-Scholes -------------------------------------------------------
def test_textbook_prices(bs_res):
    assert round(bs_res["textbook_call"], 2) == 4.76
    assert round(bs_res["textbook_put"], 2) == 0.81


def test_matches_independent_numerical_integration(bs_res):
    assert bs_res["max_abs_gap_vs_quadrature"] < 1e-9


def test_put_call_parity(bs_res):
    assert bs_res["max_parity_gap"] < 1e-10


def test_rejects_bad_option_type():
    with pytest.raises(ValueError):
        bs.price(100, 100, 1, 0.05, 0.2, option="straddle")


# ---- Binomial tree -------------------------------------------------------
def test_tree_converges_at_first_order(tree_res):
    assert -1.3 < tree_res["convergence_slope_raw"] < -0.7
    assert tree_res["abs_error"][-1] < 1e-3            # N = 2560
    assert tree_res["abs_error_smoothed"][-1] < 1e-4   # averaging N and N+1


def test_tree_put_call_parity_is_exact(tree_res):
    assert tree_res["max_parity_gap"] < 1e-9


def test_textbook_two_step_example(tree_res):
    # Hull rounds the risk-neutral probability to 0.6282, hence the 1e-3 tolerance.
    assert abs(tree_res["hull_european_put"] - 4.1923) < 1e-3
    assert abs(tree_res["hull_american_put"] - 5.0894) < 1e-3


def test_american_call_without_dividends_equals_european(tree_res):
    assert abs(tree_res["american_call_minus_european_call_no_div"]) < 1e-9


def test_early_exercise_premiums(tree_res):
    assert tree_res["american_put_early_exercise_premium"] > 0.4
    assert tree_res["american_call_premium_with_div"] > 0.0


def test_american_put_converges(tree_res):
    p = tree_res["american_put_by_steps"]
    assert abs(p[4000] - p[2000]) < abs(p[500] - p[250])


def test_tree_rejects_invalid_probability():
    with pytest.raises(ValueError):
        bt.crr_price(100, 100, 1.0, 0.5, 0.01, steps=2)   # r*dt >> sigma*sqrt(dt)


# ---- Monte Carlo ---------------------------------------------------------
def test_ci_coverage_is_near_nominal(mc_res):
    for name, r in mc_res["variants"].items():
        assert 0.92 <= r["coverage_95"] <= 0.98, name


def test_estimators_are_unbiased_within_noise(mc_res):
    for name, r in mc_res["variants"].items():
        assert abs(r["bias"]) < 3.5 * np.sqrt(r["variance"] / mc_res["reps"]), name


def test_variance_reduction(mc_res):
    vr = {k: r["variance_reduction_vs_plain"] for k, r in mc_res["variants"].items()}
    assert vr["antithetic"] > 1.5
    assert vr["control variate"] > 3.0
    assert vr["antithetic + control variate"] > 10.0


# ---- Greeks --------------------------------------------------------------
def test_finite_difference_greeks_match_analytic(greek_res):
    w = greek_res["fd_vs_analytic_worst_scaled_error"]
    assert w["delta"] < 1e-3 and w["gamma"] < 1e-3
    assert w["vega"] < 1e-4 and w["theta"] < 1e-4 and w["rho"] < 1e-4


def test_gamma_error_is_u_shaped_in_bump_size(greek_res):
    errs = np.array(greek_res["gamma_bump_errors"])
    i = int(errs.argmin())
    assert 0 < i < len(errs) - 1
    assert errs[0] > 100 * errs[i] and errs[-1] > 100 * errs[i]


def test_tree_greeks_converge(greek_res):
    t = greek_res["tree_rel_error"]
    for k in ("delta", "gamma", "theta"):
        assert t[4000][k] < t[250][k]
    assert t[4000]["delta"] < 1e-4 and t[4000]["gamma"] < 1e-3 and t[4000]["theta"] < 1e-3


def test_monte_carlo_greeks_with_common_random_numbers(greek_res):
    e = greek_res["mc_common_random_numbers_rel_error"]
    assert e["delta"] < 1e-2 and e["vega"] < 1e-2
