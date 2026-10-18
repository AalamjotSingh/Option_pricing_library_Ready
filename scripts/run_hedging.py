"""Delta-hedging experiment: hedge error and transaction costs vs rebalancing frequency.

Short one at-the-money 1-year call priced and hedged with Black-Scholes, hedge
rebalanced 1..260 times over the option's life on shared simulated GBM paths.
Run:  python scripts/run_hedging.py
"""
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from optionlab import hedging as hg  # noqa: E402

S0, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.20
N_FINE = 260                       # 52 weeks x 5 trading days
EVERY = [260, 130, 65, 20, 10, 5, 2, 1]   # fine steps between rebalances -> 1..260 hedges
COSTS = [0.0, 0.001, 0.005]        # proportional cost per unit traded notional
N_PATHS = 50_000


def slope(x, y):
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def main(seed=0, n_paths=N_PATHS):
    rng = np.random.default_rng(seed)
    paths = hg.simulate_gbm(S0, mu=R, sigma=SIGMA, T=T, n_steps=N_FINE, n_paths=n_paths, rng=rng)

    rows = []
    for c in COSTS:
        for every in EVERY:
            res = hg.delta_hedge_short_call(paths, K, T, R, SIGMA, every=every, cost_rate=c)
            row = hg.summarize(res)
            row.update(cost_rate=c, rebalance_every_fine_steps=every)
            rows.append(row)
    df = pd.DataFrame(rows)

    # Hedge with the wrong volatility (daily hedging, no costs): realised vol stays 20%.
    mis = []
    for sig_h in (0.15, 0.20, 0.25):
        res = hg.delta_hedge_short_call(paths, K, T, R, sig_h, every=1)
        s = hg.summarize(res)
        mis.append({"hedge_vol": sig_h, "mean_pnl_pct_of_premium": s["mean_pnl_pct"],
                    "std_pnl_pct_of_premium": s["std_pnl_pct"]})

    free = df[(df.cost_rate == 0.0) & (df.n_intervals >= 4)]
    summary = {
        "setup": dict(S0=S0, K=K, T=T, r=R, sigma=SIGMA, n_paths=n_paths, fine_steps=N_FINE, seed=seed),
        "std_vs_n_loglog_slope_no_costs": slope(free.n_intervals, free.std_pnl_pct),
        "interior_cost_vs_n_loglog_slope": {
            str(c): slope(d.n_intervals, d.mean_interior_cost_pct)
            for c, d in df[(df.cost_rate > 0) & (df.n_intervals >= 4)].groupby("cost_rate")},
        "theory": "hedge-error std ~ n^-0.5; interior transaction cost ~ cost_rate * n^+0.5",
        "vol_misspecification": mis,
    }

    OUT = ROOT / "results"
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "hedging_results.csv", index=False)
    (OUT / "hedging_summary.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    d0 = df[df.cost_rate == 0.0]
    ax[0].loglog(d0.n_intervals, d0.std_pnl_pct, "o-", label="simulated")
    ref = d0.std_pnl_pct.iloc[-1] * np.sqrt(d0.n_intervals.iloc[-1] / d0.n_intervals)
    ax[0].loglog(d0.n_intervals, ref, "k--", label=r"$n^{-1/2}$ reference")
    ax[0].set(xlabel="rebalances over option life (n)", ylabel="std of hedge P&L (% of premium)",
              title="Hedge error vs rebalancing frequency")
    ax[0].legend()
    for c, d in df[df.cost_rate > 0].groupby("cost_rate"):
        ax[1].loglog(d.n_intervals[1:], d.mean_interior_cost_pct[1:], "o-", label=f"cost = {c:.1%}")
    d = df[df.cost_rate == COSTS[1]]
    ref = d.mean_interior_cost_pct.iloc[-1] * np.sqrt(d.n_intervals / d.n_intervals.iloc[-1])
    ax[1].loglog(d.n_intervals[1:], ref[1:], "k--", label=r"$n^{+1/2}$ reference")
    ax[1].set(xlabel="rebalances (n)", ylabel="mean rebalancing cost (% of premium)",
              title="Transaction cost vs rebalancing frequency")
    ax[1].legend()
    for c, d in df[df.cost_rate > 0].groupby("cost_rate"):
        ax[2].plot(d.mean_cost_pct, d.std_pnl_pct, "o-", label=f"cost = {c:.1%}")
        for _, row in d.iterrows():
            ax[2].annotate(int(row.n_intervals), (row.mean_cost_pct, row.std_pnl_pct),
                           fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax[2].set(xlabel="mean total cost (% of premium)", ylabel="std of hedge P&L (% of premium)",
              title="Risk-cost trade-off (labels = n)")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(OUT / "hedging.png", dpi=140)
    return df, summary


if __name__ == "__main__":
    df, summary = main()
    pd.set_option("display.width", 160)
    print(df.round(3).to_string(index=False))
    print(json.dumps(summary, indent=2))
