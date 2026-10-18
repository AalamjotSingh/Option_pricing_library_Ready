import numpy as np
import pytest

torch = pytest.importorskip("torch")
from optionlab import surrogate as sg  # noqa: E402


@pytest.fixture(scope="module")
def trained():
    Xtr, Xva, Xte = sg.sample_domain(50_000, 1), sg.sample_domain(5_000, 2), sg.sample_domain(10_000, 3)
    torch.manual_seed(0)
    model = sg.PriceNet(128, 3)
    sg.train(model, Xtr, sg.bs_target(Xtr), Xva, sg.bs_target(Xva), epochs=30, batch_size=512, seed=0)
    return model, Xtr, Xte


def test_network_beats_baselines(trained):
    model, Xtr, Xte = trained
    y = sg.bs_target(Xte)
    nn_rmse = sg.error_metrics(sg.predict(model, Xte), y)["rmse"]
    assert nn_rmse < sg.error_metrics(sg.intrinsic_baseline(Xte), y)["rmse"] / 10
    assert nn_rmse < sg.error_metrics(sg.polynomial_baseline(Xtr, sg.bs_target(Xtr), Xte), y)["rmse"]


def test_autograd_delta_tracks_analytic_delta(trained):
    model, _, Xte = trained
    d_nn, v_nn = sg.autograd_delta_vega(model, Xte)
    d_true, _ = sg.bs_delta_vega(Xte)
    assert d_nn.shape == d_true.shape == v_nn.shape
    assert np.sqrt(np.mean((d_nn - d_true) ** 2)) < 0.1   # an untrained network is ~0.3-0.5


def test_training_is_reproducible():
    X, Xv = sg.sample_domain(4_000, 1), sg.sample_domain(1_000, 2)
    runs = []
    for _ in range(2):
        torch.manual_seed(0)
        m = sg.PriceNet(16, 2)
        h = sg.train(m, X, sg.bs_target(X), Xv, sg.bs_target(Xv), epochs=3, seed=0)
        runs.append(h)
    assert runs[0] == runs[1]
