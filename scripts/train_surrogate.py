"""Train and evaluate the PyTorch call-price surrogate.

Run:  python scripts/train_surrogate.py [--epochs 60] [--n-train 200000]
Prices are reported as C/K, so multiply by 100 for dollars per $100 of strike.
"""
import argparse
import json
import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from optionlab import surrogate as sg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-train", type=int, default=200_000)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    Xtr, Xva, Xte = (sg.sample_domain(n, s) for n, s in ((args.n_train, 1), (20_000, 2), (50_000, 3)))
    ytr, yva, yte = sg.bs_target(Xtr), sg.bs_target(Xva), sg.bs_target(Xte)

    torch.manual_seed(args.seed)
    model = sg.PriceNet(args.width, args.depth)
    n_params = sum(p.numel() for p in model.parameters())
    t0 = time.time()
    history = sg.train(model, Xtr, ytr, Xva, yva, epochs=args.epochs, seed=args.seed, log_every=5)
    train_seconds = time.time() - t0

    pred = sg.predict(model, Xte)
    hard = (Xte[:, 1] < 0.25) & (np.abs(Xte[:, 0] - 1.0) < 0.05)   # short-dated, near the money
    d_true, v_true = sg.bs_delta_vega(Xte)
    d_nn, v_nn = sg.autograd_delta_vega(model, Xte)
    metrics = {
        "setup": dict(n_train=args.n_train, n_val=len(Xva), n_test=len(Xte), epochs=args.epochs,
                      width=args.width, depth=args.depth, parameters=n_params, seed=args.seed,
                      train_seconds=round(train_seconds, 1), torch=torch.__version__,
                      domain=sg.DOMAIN),
        "price_error_C_over_K": {
            "network": sg.error_metrics(pred, yte),
            "network_short_dated_near_the_money": sg.error_metrics(pred[hard], yte[hard]),
            "polynomial_degree3_baseline": sg.error_metrics(sg.polynomial_baseline(Xtr, ytr, Xte), yte),
            "intrinsic_value_baseline": sg.error_metrics(sg.intrinsic_baseline(Xte), yte)},
        "autograd_greeks": {
            "delta_rmse": float(np.sqrt(np.mean((d_nn - d_true) ** 2))),
            "delta_max_abs": float(np.max(np.abs(d_nn - d_true))),
            "vega_rmse": float(np.sqrt(np.mean((v_nn - v_true) ** 2))),
            "vega_mean_abs_true": float(np.mean(np.abs(v_true)))},
        "no_arbitrage_violations": {
            "negative_price_fraction": float(np.mean(pred < 0)),
            "negative_delta_fraction": float(np.mean(d_nn < 0)),
            "delta_above_one_fraction": float(np.mean(d_nn > 1))},
        "validation_rmse_by_epoch": history,
    }
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "surrogate_metrics.json").write_text(json.dumps(metrics, indent=2))
    torch.save(model.state_dict(), out / "surrogate.pt")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].semilogy(range(1, len(history) + 1), history)
    ax[0].set(xlabel="epoch", ylabel="validation RMSE of C/K", title="Training curve")
    bins = np.linspace(0.7, 1.3, 25)
    which = np.digitize(Xte[:, 0], bins)
    centers, mean_e, p99_e = [], [], []
    for b in range(1, len(bins)):
        e = np.abs(pred - yte)[which == b]
        if len(e):
            centers.append(0.5 * (bins[b - 1] + bins[b]))
            mean_e.append(e.mean())
            p99_e.append(np.quantile(e, 0.99))
    ax[1].semilogy(centers, mean_e, "o-", label="mean")
    ax[1].semilogy(centers, p99_e, "s-", label="99th percentile")
    ax[1].set(xlabel="moneyness S/K", ylabel="|price error| (C/K)", title="Test error by moneyness")
    ax[1].legend()
    sel = np.random.default_rng(0).choice(len(Xte), 4000, replace=False)
    ax[2].scatter(d_true[sel], d_nn[sel], s=3, alpha=0.4)
    ax[2].plot([0, 1], [0, 1], "k--", lw=1)
    ax[2].set(xlabel="analytic delta", ylabel="autograd delta of network", title="Delta from autograd")
    fig.tight_layout()
    fig.savefig(out / "surrogate.png", dpi=140)
    print(json.dumps({k: v for k, v in metrics.items() if k != "validation_rmse_by_epoch"}, indent=2))


if __name__ == "__main__":
    main()
