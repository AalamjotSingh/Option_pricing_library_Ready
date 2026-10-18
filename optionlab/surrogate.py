"""Neural-network surrogate for Black-Scholes European call prices (PyTorch).

The call price is homogeneous of degree one in (S, K), so the network learns
y = C / K as a function of moneyness m = S / K, maturity T, rate r and volatility
sigma. Autograd then gives delta (= dy/dm) and vega (= K * dy/dsigma) from the
same network, which is checked against the closed-form Greeks.
"""
from __future__ import annotations

import math
from itertools import combinations_with_replacement

import numpy as np
import torch
from torch import nn

from . import black_scholes as bs

FEATURES = ("m", "T", "r", "sigma")
DOMAIN = {"m": (0.7, 1.3), "T": (0.05, 2.0), "r": (0.0, 0.08), "sigma": (0.10, 0.60)}


def sample_domain(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lo = np.array([DOMAIN[k][0] for k in FEATURES])
    hi = np.array([DOMAIN[k][1] for k in FEATURES])
    return lo + (hi - lo) * rng.random((n, len(FEATURES)))


def bs_target(X: np.ndarray) -> np.ndarray:
    """Closed-form C/K (set K = 1, S = m)."""
    return np.asarray(bs.price(X[:, 0], 1.0, X[:, 1], X[:, 2], X[:, 3]))


def bs_delta_vega(X: np.ndarray):
    g = bs.greeks(X[:, 0], 1.0, X[:, 1], X[:, 2], X[:, 3])
    return np.asarray(g["delta"]), np.asarray(g["vega"])


class PriceNet(nn.Module):
    """MLP on inputs rescaled to [-1, 1]; takes raw (m, T, r, sigma), returns C/K."""

    def __init__(self, width: int = 128, depth: int = 3):
        super().__init__()
        lo = torch.tensor([DOMAIN[k][0] for k in FEATURES], dtype=torch.float32)
        hi = torch.tensor([DOMAIN[k][1] for k in FEATURES], dtype=torch.float32)
        self.register_buffer("lo", lo)
        self.register_buffer("hi", hi)
        layers, d_in = [], len(FEATURES)
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), nn.SiLU()]   # smooth activation -> smooth Greeks
            d_in = width
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(2.0 * (x - self.lo) / (self.hi - self.lo) - 1.0).squeeze(-1)


def train(model, X_train, y_train, X_val, y_val, epochs=60, batch_size=2048, lr=3e-3, seed=0,
          log_every=0):
    """Mean-squared-error training with Adam and a one-cycle learning-rate schedule.

    Returns the validation RMSE after each epoch.
    """
    torch.manual_seed(seed)
    Xt = torch.as_tensor(X_train, dtype=torch.float32)
    yt = torch.as_tensor(y_train, dtype=torch.float32)
    Xv = torch.as_tensor(X_val, dtype=torch.float32)
    yv = torch.as_tensor(y_val, dtype=torch.float32)
    n = len(Xt)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * math.ceil(n / batch_size))
    gen = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            loss = torch.mean((model(Xt[idx]) - yt[idx]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            history.append(float(torch.sqrt(torch.mean((model(Xv) - yv) ** 2))))
        if log_every and (epoch + 1) % log_every == 0:
            print(f"epoch {epoch + 1}/{epochs} val_rmse={history[-1]:.2e}", flush=True)
    return history


def predict(model, X) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(X, dtype=torch.float32)).numpy().astype(np.float64)


def autograd_delta_vega(model, X):
    """Delta and vega of the learned price surface via autograd (K = 1)."""
    model.eval()
    x = torch.as_tensor(X, dtype=torch.float32).clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(model(x).sum(), x)
    g = grad.numpy().astype(np.float64)
    return g[:, 0], g[:, 3]


def polynomial_baseline(X_train, y_train, X_test, degree: int = 3) -> np.ndarray:
    """Least-squares polynomial regression on the raw inputs (no sklearn needed)."""
    def feats(X):
        cols = [np.ones(len(X))]
        for d in range(1, degree + 1):
            for combo in combinations_with_replacement(range(X.shape[1]), d):
                cols.append(np.prod(X[:, combo], axis=1))
        return np.stack(cols, axis=1)

    coef, *_ = np.linalg.lstsq(feats(X_train), y_train, rcond=None)
    return feats(X_test) @ coef


def intrinsic_baseline(X) -> np.ndarray:
    """Discounted intrinsic value max(m - exp(-rT), 0): the 'no model' baseline."""
    return np.maximum(X[:, 0] - np.exp(-X[:, 2] * X[:, 1]), 0.0)


def error_metrics(pred, true) -> dict:
    err = np.abs(np.asarray(pred) - np.asarray(true))
    return {"rmse": float(np.sqrt(np.mean(err**2))), "mae": float(err.mean()),
            "p99_abs": float(np.quantile(err, 0.99)), "max_abs": float(err.max())}
