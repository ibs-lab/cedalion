import numpy as np
import xarray as xr

from cedalion.sigdecomp.multimodal.tcca import tCCA, temporal_embedding


def test_tcca_optimal_shift_is_argmax_of_source_correlation():
    """optimal_shift is the shift maximizing |corr(sx_i, sy)| of the fitted sources.

    Regression test: estimate_optimal_shift standardized the already standardized
    data a second time (with the mean and std of the raw data), which distorted the
    correlations for features of different scales and picked a neighbouring shift
    in ambiguous cases.
    """
    fs = 10.0
    nt, nx, ny = 1500, 6, 4
    t = np.arange(nt) / fs
    shifts = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    rng = np.random.default_rng(12)
    true_lag = rng.choice([0.3, 0.5, 0.7])
    k = int(round(true_lag * fs))
    s = np.convolve(rng.standard_normal(nt + k), np.ones(4) / 4, "same")
    ax, ay = rng.standard_normal(nx), rng.standard_normal(ny)
    # features of very different scales
    x_scales = 10.0 ** rng.uniform(-1.5, 1.5, nx)
    x = (np.outer(s[k : k + nt], ax) + 0.5 * rng.standard_normal((nt, nx))) * x_scales
    y = np.outer(s[:nt], ay) + 0.5 * rng.standard_normal((nt, ny))

    X = xr.DataArray(
        x,
        dims=("time", "channel"),
        coords={"time": t, "channel": [f"x{i}" for i in range(nx)]},
    )
    Y = xr.DataArray(
        y,
        dims=("time", "channel"),
        coords={"time": t, "channel": [f"y{i}" for i in range(ny)]},
    )

    np.random.seed(12)
    model = tCCA(N_components=1, time_shifts=shifts)
    model.fit(X, Y)

    X_emb = temporal_embedding(X, model.time_shifts)
    X_n = (X_emb - model.x_mean) / model.x_std
    Y_n = (Y - model.y_mean) / model.y_std
    sy = (Y_n @ model.Wy)[:, 0]
    corrs = [
        abs(np.corrcoef((X_n[i] @ model.Wx[i])[:, 0], sy)[0, 1])
        for i in range(model.N_shifts)
    ]

    assert model.optimal_shift[0] == model.time_shifts[int(np.argmax(corrs))]
