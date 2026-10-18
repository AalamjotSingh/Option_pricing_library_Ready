import numpy as np
import pytest

from optionlab import hedging as hg

S0, K, T, R, SIGMA, N_FINE = 100.0, 100.0, 1.0, 0.05, 0.20, 260


@pytest.fixture(scope="module")
def paths():
    rng = np.random.default_rng(1)
    return hg.simulate_gbm(S0, R, SIGMA, T, N_FINE, 20_000, rng)


def test_paths_have_risk_neutral_mean(paths):
    se = paths[:, -1].std() / np.sqrt(paths.shape[0])
    assert abs(paths[:, -1].mean() - S0 * np.exp(R * T)) < 4 * se


def test_hedge_error_shrinks_like_inverse_sqrt_n(paths):
    ns, stds = [], []
    for every in (65, 20, 10, 5, 2, 1):
        res = hg.delta_hedge_short_call(paths, K, T, R, SIGMA, every=every)
        ns.append(res.n_intervals)
        stds.append(res.pnl.std(ddof=1))
    assert all(a > b for a, b in zip(stds, stds[1:]))
    slope = np.polyfit(np.log(ns), np.log(stds), 1)[0]
    assert -0.62 < slope < -0.38


def test_frictionless_daily_hedge_is_unbiased_and_tight(paths):
    res = hg.delta_hedge_short_call(paths, K, T, R, SIGMA, every=1)
    se = res.pnl.std(ddof=1) / np.sqrt(len(res.pnl))
    assert abs(res.pnl.mean()) < 4 * se
    assert res.pnl.std(ddof=1) / res.premium < 0.06


def test_costs_reduce_pnl_by_exactly_their_present_value(paths):
    free = hg.delta_hedge_short_call(paths, K, T, R, SIGMA, every=5)
    costly = hg.delta_hedge_short_call(paths, K, T, R, SIGMA, every=5, cost_rate=0.005)
    assert np.allclose(costly.pnl, free.pnl - costly.cost, atol=1e-10)


def test_more_rebalancing_costs_more(paths):
    costs = [hg.delta_hedge_short_call(paths, K, T, R, SIGMA, every=e, cost_rate=0.001)
             .interior_cost.mean() for e in (65, 20, 5, 1)]
    assert costs == sorted(costs)


def test_wrong_volatility_shifts_mean_pnl_in_the_expected_direction(paths):
    low = hg.delta_hedge_short_call(paths, K, T, R, 0.15, every=1).pnl.mean()
    high = hg.delta_hedge_short_call(paths, K, T, R, 0.25, every=1).pnl.mean()
    assert low < 0 < high   # sold the option too cheap / too dear relative to realised vol


def test_rebalance_interval_must_divide_grid(paths):
    with pytest.raises(ValueError):
        hg.delta_hedge_short_call(paths, K, T, R, SIGMA, every=7)
