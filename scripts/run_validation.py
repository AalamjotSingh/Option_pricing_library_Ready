"""Run every validation check and write a report plus figure to results/.

Run:  python scripts/run_validation.py
"""
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from optionlab import validation as v  # noqa: E402


def report(res) -> str:
    b, t, m, g = res["black_scholes"], res["tree"], res["monte_carlo"], res["greeks"]
    L = ["# Validation report", "",
         "Produced by `python scripts/run_validation.py` (seed 0). Base case: S=K=100, T=1, r=5%, "
         "sigma=20% unless stated.", "",
         "## 1. Black-Scholes closed form", "", "| check | result |", "|---|---|",
         f"| Textbook call / put (S=42, K=40, T=0.5, r=10%, sigma=20%) | {b['textbook_call']:.4f} / "
         f"{b['textbook_put']:.4f} (book: 4.76 / 0.81) |",
         f"| Max abs gap vs numerical integration of the risk-neutral expectation "
         f"({b['n_parameter_sets']} random parameter sets, calls and puts, with dividends) | "
         f"{b['max_abs_gap_vs_quadrature']:.1e} |",
         f"| Max put-call parity gap | {b['max_parity_gap']:.1e} |", "",
         "## 2. Binomial tree (Cox-Ross-Rubinstein)", "", "| check | result |", "|---|---|",
         f"| European call error at N=10 / 160 / 2560 steps | {t['abs_error'][0]:.2e} / "
         f"{t['abs_error'][4]:.2e} / {t['abs_error'][-1]:.2e} |",
         f"| Log-log convergence slope (raw / averaging N and N+1) | {t['convergence_slope_raw']:.2f} / "
         f"{t['convergence_slope_smoothed']:.2f} (first order = -1) |",
         f"| Error at N=2560 after averaging N and N+1 | {t['abs_error_smoothed'][-1]:.1e} |",
         f"| Max put-call parity gap inside the tree | {t['max_parity_gap']:.1e} |",
         f"| Hull two-step example, European / American put | {t['hull_european_put']:.4f} / "
         f"{t['hull_american_put']:.4f} (book: 4.1923 / 5.0894; book rounds p to 4 d.p.) |",
         f"| American minus European call, no dividends | {t['american_call_minus_european_call_no_div']:.1e} |",
         f"| American call premium, dividend yield 8% (K=80) | {t['american_call_premium_with_div']:.3f} |",
         f"| American put, N = 250 / 1000 / 4000 | {t['american_put_by_steps'][250]:.4f} / "
         f"{t['american_put_by_steps'][1000]:.4f} / {t['american_put_by_steps'][4000]:.4f} |",
         f"| Early-exercise premium of the ATM put (N=4000) | {t['american_put_early_exercise_premium']:.4f} |", "",
         f"## 3. Monte Carlo ({m['reps']} replications x {m['n_paths']:,} paths, ATM call, "
         f"exact price {m['exact']:.4f})", "",
         "| variant | 95% CI coverage | RMSE | variance reduction vs plain |", "|---|---|---|---|"]
    for name, r in m["variants"].items():
        L.append(f"| {name} | {r['coverage_95']:.1%} | {r['rmse']:.4f} | {r['variance_reduction_vs_plain']:.1f}x |")
    w = g["fd_vs_analytic_worst_scaled_error"]
    L += ["", "## 4. Greeks", "",
          "Scaled error = |numerical - analytic| / max(|analytic|, 0.01).", "",
          "| method | delta | gamma | vega | theta | rho |", "|---|---|---|---|---|---|",
          "| Finite difference on closed form, worst over 40 random sets x call/put | "
          + " | ".join(f"{w[k]:.1e}" for k in ("delta", "gamma", "vega", "theta", "rho")) + " |"]
    for n, e in g["tree_rel_error"].items():
        L.append(f"| CRR tree Greeks, N={n} (relative error, base case) | {e['delta']:.1e} | "
                 f"{e['gamma']:.1e} | - | {e['theta']:.1e} | - |")
    e = g["mc_common_random_numbers_rel_error"]
    L.append(f"| Monte Carlo bump-and-revalue, common random numbers, 400k paths | {e['delta']:.1e} | - | "
             f"{e['vega']:.1e} | - | - |")
    bumps, errs = np.array(g["gamma_bump_sizes"]), np.array(g["gamma_bump_errors"])
    i = int(errs.argmin())
    L += ["", f"Gamma by finite difference is U-shaped in the bump size: relative error {errs[0]:.1e} at a "
          f"bump of {bumps[0]:.0e} x S (round-off), minimum {errs[i]:.1e} at {bumps[i]:.0e} x S, "
          f"{errs[-1]:.1e} at {bumps[-1]:.0e} x S (truncation).", ""]
    return "\n".join(L)


def figure(res, path):
    t, m, g = res["tree"], res["monte_carlo"], res["greeks"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    n = np.array(t["steps"], float)
    ax[0].loglog(n, t["abs_error"], "o-", label="CRR tree")
    ax[0].loglog(n, t["abs_error_smoothed"], "s-", label="average of N and N+1")
    ax[0].loglog(n, t["abs_error"][0] * n[0] / n, "k--", label="1/N reference")
    ax[0].set(xlabel="tree steps N", ylabel="|tree - Black-Scholes|", title="European call: tree convergence")
    ax[0].legend()
    ax[1].loglog(g["gamma_bump_sizes"], g["gamma_bump_errors"], "o-")
    ax[1].set(xlabel="bump size (fraction of S)", ylabel="relative gamma error",
              title="Finite-difference gamma: round-off vs truncation")
    names = list(m["variants"])
    vr = [m["variants"][k]["variance_reduction_vs_plain"] for k in names]
    ax[2].bar(range(len(names)), vr)
    ax[2].set_xticks(range(len(names)), [k.replace(" + ", "\n+ ").replace(" ", "\n", 1) if len(k) < 16 else
                                         k.replace(" + ", "\n+ ") for k in names], fontsize=8)
    for i, (k, val) in enumerate(zip(names, vr)):
        ax[2].text(i, val, f"{val:.1f}x\ncov {m['variants'][k]['coverage_95']:.0%}", ha="center", va="bottom", fontsize=8)
    ax[2].set(ylabel="variance reduction vs plain MC", title="Monte Carlo variance reduction", ylim=(0, max(vr) * 1.25))
    fig.tight_layout()
    fig.savefig(path, dpi=140)


if __name__ == "__main__":
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    res = v.run_all(seed=0)
    (out / "validation.json").write_text(json.dumps(res, indent=2, default=float))
    md = report(res)
    (out / "validation_report.md").write_text(md)
    figure(res, out / "validation.png")
    print(md)
