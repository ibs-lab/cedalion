import cedalion.data
import cedalion.dot as dot
import cedalion.nirs as nirs
import cedalion.typing as cdt
import numpy as np
import xarray as xr
import cedalion.xrutils as xrutils
import pytest
from dataclasses import dataclass
from scipy.sparse import csr_array




@dataclass
class ReconData:
    Adot: xr.DataArray
    x_true : cdt.NDTimeSeries
    y : cdt.NDTimeSeries
    c_meas : cdt.NDTimeSeries

@pytest.fixture
def recon_data():
    rng = np.random.default_rng(seed=42)

    nchannel = 50
    nvertex = 20
    nwavelength = 2

    channel = [f"C{c:02d}" for c in range(1, nchannel + 1)]
    wavelength = [760.0, 850.0]

    Adot = xr.DataArray(
        rng.uniform(0.1, 20.0, size=(nchannel, nvertex, nwavelength)),
        dims=["channel", "vertex", "wavelength"],
        coords={
            "is_brain": ("vertex", [True] * 15 + [False] * 5),
            "channel": channel,
            "source": ("channel", [f"S{c:02d}" for c in range(1, nchannel + 1)]),
            "detector": ("channel", [f"D{c:02d}" for c in range(1, nchannel + 1)]),
            "wavelength": wavelength,
        },
    )

    x_true = xr.DataArray(
        np.zeros(Adot.sizes["vertex"]),
        dims=["vertex"],
        coords={"is_brain": ("vertex", Adot.is_brain.values)},
    )

    x_true[5:15] = 1.0

    y = xrutils.contract(Adot, x_true, "vertex")

    c_meas = xr.DataArray(
        np.ones((nchannel, nwavelength)),
        dims=["channel", "wavelength"],
        coords={
            "channel": channel,
            "wavelength": wavelength,
        },
    )

    return ReconData(Adot, x_true, y, c_meas)


# FIXME mock implementation for spatial basis functions
@pytest.mark.parametrize("lambda_R_conc", [None, 1.])
@pytest.mark.parametrize("apply_c_meas", [False, True])
@pytest.mark.parametrize("alpha_spatial", [None, 1.])
@pytest.mark.parametrize("brain_only", [False, True])
@pytest.mark.parametrize("recon_mode", ["mua", "conc", "mua2conc"])
def test_image_recon(
    recon_data: ReconData,
    recon_mode,
    brain_only,
    alpha_spatial,
    apply_c_meas,
    lambda_R_conc,
):

    xrutils.unit_stripping_is_error(True)

    print(
        f"{recon_mode=} {brain_only=} {alpha_spatial=} {apply_c_meas=} {lambda_R_conc=}"
    )

    recon = dot.ImageRecon(
        recon_data.Adot,
        recon_mode=recon_mode,
        brain_only=brain_only,
        spatial_basis_functions=None,
        alpha_meas=1e-6,
        alpha_spatial=alpha_spatial,
        apply_c_meas=apply_c_meas,
        lambda_R_conc= lambda_R_conc,
    )

    if apply_c_meas:
        c_meas = recon_data.c_meas
    else:
        c_meas = None

    x_reco = recon.reconstruct(recon_data.y, c_meas)

    if brain_only:
        nv_brain = np.sum(recon_data.x_true.is_brain.values)
        assert x_reco.shape == (2, nv_brain)
    else:
        assert x_reco.shape == (2, len(recon_data.x_true))


    assert not np.iscomplex(x_reco.data).any()


# The tests below compare ImageRecon against an independent dense numpy reference.
# Whatever the option, the problem is y = M u with unknowns u, prior R and image
# x = P u:
#
#   W = R M^T inv(M R M^T + lambda_meas C)
#
# In recon_mode 'conc' there is one M with rows (wavelength, channel) and columns
# (chromo, unknown). In the other modes there is one M per wavelength.

ALPHA_MEAS = 1e-3
N_BRAIN, N_SCALP = 30, 10
N_KERNEL_BRAIN, N_KERNEL_SCALP = 6, 2
N_CHANNEL = 12
N_TIME = 3

MODES = ["conc", "mua", "mua2conc"]


@dataclass
class DenseCase:
    Adot: xr.DataArray
    G: np.ndarray  # (kernel, vertex)
    sbf: dot.GaussianSpatialBasisFunctions
    y: cdt.NDTimeSeries
    c_meas: xr.DataArray
    E: np.ndarray  # (wavelength, chromo)


def make_kernels(rng, n_kernel, n_vertex, width=5):
    """Overlapping kernels. The overlap is needed for correlated kernel weights."""
    G = np.zeros((n_kernel, n_vertex))
    step = n_vertex // n_kernel
    for k in range(n_kernel):
        start, stop = max(0, k * step - width), min(n_vertex, (k + 1) * step + width)
        G[k, start:stop] = rng.uniform(0.5, 1.5, stop - start)
    return G


def make_sbf(Adot, G):
    """Gaussian SBFs with the given kernel values, bypassing the head model."""
    n_kernel, n_vertex = G.shape

    sbf = dot.GaussianSpatialBasisFunctions.__new__(dot.GaussianSpatialBasisFunctions)
    sbf._G = csr_array(G)
    sbf._G_kernel = np.arange(n_kernel)
    sbf._G_kernel_is_brain = np.arange(n_kernel) < N_KERNEL_BRAIN
    sbf._G_vertex_is_brain = np.arange(n_vertex) < N_BRAIN
    sbf._G_vertex_parcel = None
    sbf._compute_H(Adot)
    return sbf


@pytest.fixture
def dense_case():
    rng = np.random.default_rng(seed=7)

    channel = [f"C{c:02d}" for c in range(N_CHANNEL)]
    wavelength = [760.0, 850.0]

    # as on a real head the scalp dominates the sensitivity
    values = rng.uniform(0.1, 20.0, size=(N_CHANNEL, N_BRAIN + N_SCALP, 2))
    values[:, N_BRAIN:, :] *= 5.0

    Adot = xr.DataArray(
        values,
        dims=["channel", "vertex", "wavelength"],
        coords={
            "is_brain": ("vertex", [True] * N_BRAIN + [False] * N_SCALP),
            "channel": channel,
            "source": ("channel", [f"S{c:02d}" for c in range(N_CHANNEL)]),
            "detector": ("channel", [f"D{c:02d}" for c in range(N_CHANNEL)]),
            "wavelength": wavelength,
        },
    )

    G = np.zeros((N_KERNEL_BRAIN + N_KERNEL_SCALP, N_BRAIN + N_SCALP))
    G[:N_KERNEL_BRAIN, :N_BRAIN] = make_kernels(rng, N_KERNEL_BRAIN, N_BRAIN)
    G[N_KERNEL_BRAIN:, N_BRAIN:] = make_kernels(rng, N_KERNEL_SCALP, N_SCALP)

    y = xr.DataArray(
        rng.normal(size=(N_CHANNEL, 2, N_TIME)),
        dims=["channel", "wavelength", "time"],
        coords={
            "channel": channel,
            "source": ("channel", Adot.source.values),
            "detector": ("channel", Adot.detector.values),
            "wavelength": wavelength,
            "time": np.arange(N_TIME, dtype=float),
        },
    )

    c_meas = xr.DataArray(
        rng.uniform(0.5, 2.0, size=(N_CHANNEL, 2)),
        dims=["channel", "wavelength"],
        coords={"channel": channel, "wavelength": wavelength},
    )

    E = nirs.get_extinction_coefficients("prahl", Adot.wavelength)
    E = E.pint.dequantify().sel(chromo=["HbO", "HbR"])
    E = E.transpose("wavelength", "chromo").values

    return DenseCase(Adot, G, make_sbf(Adot, G), y, c_meas, E)


def reference_basis(Adot, G, use_sbf, brain_only):
    """Sensitivity S (channel, unknown, wavelength) and P (vertex, unknown)."""
    n_vertex = N_BRAIN + N_SCALP
    S = Adot.values
    P = np.eye(n_vertex)
    unknown_is_brain = np.arange(n_vertex) < N_BRAIN

    if use_sbf:
        S = np.einsum("cvw,kv->ckw", S, G)
        P = G.T
        unknown_is_brain = np.arange(G.shape[0]) < N_KERNEL_BRAIN

    if brain_only:
        S = S[:, unknown_is_brain, :]
        P = P[:N_BRAIN, unknown_is_brain]

    return S, P


def reference_matrices(case: DenseCase, S, recon_mode):
    if recon_mode == "conc":
        rows = [
            np.hstack([case.E[w, c] * S[:, :, w] for c in range(case.E.shape[1])])
            for w in range(S.shape[2])
        ]
        return [np.vstack(rows)]
    return [S[:, :, w] for w in range(S.shape[2])]


def reference_rows(values, recon_mode):
    """Bring an array with leading dims (channel, wavelength) into solver order."""
    if recon_mode == "conc":
        return [np.concatenate([values[:, w] for w in range(values.shape[1])])]
    return [values[:, w] for w in range(values.shape[1])]


def reference_prior(M, alpha_spatial):
    if alpha_spatial is None:
        return np.ones(M.shape[1])
    B = np.sum(M**2, axis=0)
    return 1 / (B + alpha_spatial * B.max())


def reference_W(M, alpha_spatial, c=None):
    R = reference_prior(M, alpha_spatial)
    F = (M * R) @ M.T
    lambda_meas = ALPHA_MEAS * np.max(np.linalg.eigvalsh(F))
    C = np.eye(len(F)) if c is None else np.diag(c)
    return (R[:, None] * M.T) @ np.linalg.inv(F + lambda_meas * C)


def reference_image(u, P, recon_mode, E):
    """Map unknowns per solver problem to image space."""
    if recon_mode == "conc":
        return np.stack([P @ part for part in np.split(u[0], E.shape[1])])

    image = np.stack([P @ part for part in u])
    if recon_mode == "mua2conc":
        return np.einsum("cw,w...->c...", np.linalg.pinv(E), image)
    return image


def reference_image_variance(cov, P, recon_mode, E):
    """Map covariances of unknowns per solver problem to image space variances."""

    def variance(C):
        return np.einsum("vi,ij,vj->v", P, C, P)

    if recon_mode == "conc":
        n_unknown = P.shape[1]
        return np.stack(
            [
                variance(cov[0][i * n_unknown:(i + 1) * n_unknown,
                                i * n_unknown:(i + 1) * n_unknown])
                for i in range(E.shape[1])
            ]
        )

    image = np.stack([variance(C) for C in cov])
    if recon_mode == "mua2conc":
        return np.linalg.pinv(E) ** 2 @ image
    return image


def assert_matches(actual, expected):
    np.testing.assert_allclose(
        actual, expected, rtol=1e-6, atol=1e-8 * np.abs(expected).max()
    )


def image_values(image, recon_mode, spatial_dims=("vertex",)):
    """Return the magnitude of an image in SI-like units with a fixed dim order."""
    is_conc = recon_mode in ["conc", "mua2conc"]
    first_dim = "chromo" if is_conc else "wavelength"

    if image.pint.units is not None:
        image = image.pint.to("M" if is_conc else "1/mm").pint.dequantify()

    return image.transpose(first_dim, *spatial_dims).values


def variance_values(image, recon_mode):
    is_conc = recon_mode in ["conc", "mua2conc"]
    first_dim = "chromo" if is_conc else "wavelength"

    if image.pint.units is not None:
        image = image.pint.to("M**2").pint.dequantify()

    return image.transpose(first_dim, "vertex").values


def xfail(request, reason):
    request.applymarker(pytest.mark.xfail(strict=True, reason=reason))


REASON_LINEAR_G = "SBF variances are mapped with G instead of diag(G cov G^T)"
REASON_PRIOR = "posterior uses R=1/(B+1) but the reconstruction R=1"
REASON_POST_DIMS = "posterior has dims (vertex, chromo) with SBFs in conc mode"


@pytest.mark.parametrize("apply_c_meas", [False, True])
@pytest.mark.parametrize("alpha_spatial", [None, 1e-2])
@pytest.mark.parametrize("brain_only", [False, True])
@pytest.mark.parametrize("use_sbf", [False, True])
@pytest.mark.parametrize("recon_mode", MODES)
def test_reconstruct_matches_dense_reference(
    dense_case: DenseCase, recon_mode, use_sbf, brain_only, alpha_spatial, apply_c_meas
):
    case = dense_case

    recon = dot.ImageRecon(
        case.Adot,
        recon_mode=recon_mode,
        brain_only=brain_only,
        spatial_basis_functions=case.sbf if use_sbf else None,
        alpha_meas=ALPHA_MEAS,
        alpha_spatial=alpha_spatial,
        apply_c_meas=apply_c_meas,
    )
    actual = recon.reconstruct(case.y, case.c_meas if apply_c_meas else None)

    S, P = reference_basis(case.Adot, case.G, use_sbf, brain_only)
    matrices = reference_matrices(case, S, recon_mode)
    y_rows = reference_rows(case.y.values, recon_mode)
    c_rows = reference_rows(case.c_meas.values, recon_mode)

    u = [
        reference_W(M, alpha_spatial, c if apply_c_meas else None) @ y
        for M, y, c in zip(matrices, y_rows, c_rows)
    ]
    expected = reference_image(u, P, recon_mode, case.E)

    assert_matches(image_values(actual, recon_mode, ("vertex", "time")), expected)


@pytest.mark.parametrize("alpha_spatial", [None, 1e-2])
@pytest.mark.parametrize("brain_only", [False, True])
@pytest.mark.parametrize("use_sbf", [False, True])
@pytest.mark.parametrize("recon_mode", MODES)
def test_image_noise_matches_dense_reference(
    request, dense_case: DenseCase, recon_mode, use_sbf, brain_only, alpha_spatial
):
    if use_sbf:
        xfail(request, REASON_LINEAR_G)

    case = dense_case

    recon = dot.ImageRecon(
        case.Adot,
        recon_mode=recon_mode,
        brain_only=brain_only,
        spatial_basis_functions=case.sbf if use_sbf else None,
        alpha_meas=ALPHA_MEAS,
        alpha_spatial=alpha_spatial,
        apply_c_meas=True,
    )
    actual = recon.get_image_noise(case.c_meas)

    S, P = reference_basis(case.Adot, case.G, use_sbf, brain_only)
    matrices = reference_matrices(case, S, recon_mode)
    c_rows = reference_rows(case.c_meas.values, recon_mode)

    cov = []
    for M, c in zip(matrices, c_rows):
        W = reference_W(M, alpha_spatial, c)
        cov.append(W @ np.diag(c) @ W.T)
    expected = reference_image_variance(cov, P, recon_mode, case.E)

    assert_matches(variance_values(actual, recon_mode), expected)


@pytest.mark.parametrize("lambda_R_conc", [None, 1.0])
@pytest.mark.parametrize("alpha_spatial", [None, 1e-2])
@pytest.mark.parametrize("brain_only", [False, True])
@pytest.mark.parametrize("use_sbf", [False, True])
@pytest.mark.parametrize("recon_mode", MODES)
def test_posterior_matches_dense_reference(
    request,
    dense_case: DenseCase,
    recon_mode,
    use_sbf,
    brain_only,
    alpha_spatial,
    lambda_R_conc,
):
    if alpha_spatial is None:
        xfail(request, REASON_PRIOR)
    elif use_sbf:
        xfail(request, REASON_LINEAR_G)

    case = dense_case

    recon = dot.ImageRecon(
        case.Adot,
        recon_mode=recon_mode,
        brain_only=brain_only,
        spatial_basis_functions=case.sbf if use_sbf else None,
        alpha_meas=ALPHA_MEAS,
        alpha_spatial=alpha_spatial,
        lambda_R_conc=lambda_R_conc,
    )
    actual = recon.get_image_noise_posterior()

    S, P = reference_basis(case.Adot, case.G, use_sbf, brain_only)
    matrices = reference_matrices(case, S, recon_mode)

    if recon_mode == "conc":
        lambda_R = [1.0 if lambda_R_conc is None else lambda_R_conc]
    else:
        lambda_R = recon.compute_lambda_R_indirect().values

    cov = []
    for M, lam in zip(matrices, lambda_R):
        R = reference_prior(M, alpha_spatial)
        W = reference_W(M, alpha_spatial)
        cov.append(lam * (np.diag(R) - W @ (M * R)))
    expected = reference_image_variance(cov, P, recon_mode, case.E)

    assert_matches(variance_values(actual, recon_mode), expected)


@pytest.mark.parametrize("recon_mode", MODES)
def test_channel_order_of_y_is_irrelevant(dense_case: DenseCase, recon_mode):
    case = dense_case
    kwargs = dict(recon_mode=recon_mode, alpha_meas=ALPHA_MEAS)

    expected = dot.ImageRecon(case.Adot, **kwargs).reconstruct(case.y)
    actual = dot.ImageRecon(case.Adot, **kwargs).reconstruct(
        case.y.isel(channel=slice(None, None, -1))
    )

    assert_matches(
        image_values(actual, recon_mode, ("vertex", "time")),
        image_values(expected, recon_mode, ("vertex", "time")),
    )


@pytest.mark.parametrize("recon_mode", MODES)
def test_pruned_channels_are_matched_by_label(dense_case: DenseCase, recon_mode):
    case = dense_case
    n_pruned = 3

    recon = dot.ImageRecon(case.Adot, recon_mode=recon_mode, alpha_meas=ALPHA_MEAS)
    actual = recon.reconstruct(case.y.isel(channel=slice(n_pruned, None)))

    # pruning keeps W, which was calculated from all channels, and drops the columns
    S, P = reference_basis(case.Adot, case.G, False, False)
    matrices = reference_matrices(case, S, recon_mode)

    y = case.y.values.copy()
    y[:n_pruned] = 0.0
    u = [
        reference_W(M, None) @ y_row
        for M, y_row in zip(matrices, reference_rows(y, recon_mode))
    ]
    expected = reference_image(u, P, recon_mode, case.E)

    assert_matches(image_values(actual, recon_mode, ("vertex", "time")), expected)


@pytest.mark.parametrize("recon_mode", MODES)
def test_channel_order_of_c_meas_is_irrelevant(dense_case: DenseCase, recon_mode):
    case = dense_case
    kwargs = dict(recon_mode=recon_mode, alpha_meas=ALPHA_MEAS, apply_c_meas=True)

    expected = dot.ImageRecon(case.Adot, **kwargs).reconstruct(case.y, case.c_meas)
    actual = dot.ImageRecon(case.Adot, **kwargs).reconstruct(
        case.y, case.c_meas.isel(channel=slice(None, None, -1))
    )

    assert_matches(
        image_values(actual, recon_mode, ("vertex", "time")),
        image_values(expected, recon_mode, ("vertex", "time")),
    )


@pytest.mark.parametrize("recon_mode", MODES)
def test_pseudoinverse_follows_c_meas(dense_case: DenseCase, recon_mode):
    case = dense_case
    kwargs = dict(recon_mode=recon_mode, alpha_meas=ALPHA_MEAS, apply_c_meas=True)
    c_other = case.c_meas * 3 + 1

    recon = dot.ImageRecon(case.Adot, **kwargs)
    recon.reconstruct(case.y, case.c_meas)
    actual = recon.reconstruct(case.y, c_other)

    expected = dot.ImageRecon(case.Adot, **kwargs).reconstruct(case.y, c_other)

    assert_matches(
        image_values(actual, recon_mode, ("vertex", "time")),
        image_values(expected, recon_mode, ("vertex", "time")),
    )


@pytest.mark.parametrize("use_sbf", [False, True])
@pytest.mark.parametrize("recon_mode", MODES)
def test_image_dims_do_not_depend_on_sbf(
    request, dense_case: DenseCase, recon_mode, use_sbf
):
    case = dense_case
    first_dim = "wavelength" if recon_mode == "mua" else "chromo"

    recon = dot.ImageRecon(
        case.Adot,
        recon_mode=recon_mode,
        spatial_basis_functions=case.sbf if use_sbf else None,
        alpha_meas=ALPHA_MEAS,
        alpha_spatial=1e-2,
        lambda_R_conc=1.0,
        apply_c_meas=True,
    )

    image = recon.reconstruct(case.y, case.c_meas)
    assert set(image.dims) == {first_dim, "vertex", "time"}
    assert recon.get_image_noise(case.c_meas).dims == (first_dim, "vertex")

    if recon_mode == "conc" and use_sbf:
        xfail(request, REASON_POST_DIMS)
    assert recon.get_image_noise_posterior(case.c_meas).dims == (first_dim, "vertex")


# Probes in which not all channels share the same wavelengths come with a list of
# sensitivities and time series, one per group of channels with identical wavelengths.
# 850 nm is measured by both groups.

MIXED_WAVELENGTHS = [[735.0, 850.0], [760.0, 850.0]]
MIXED_CHANNELS = [[f"A{c:02d}" for c in range(7)], [f"B{c:02d}" for c in range(8)]]


@dataclass
class MixedCase:
    Adots: list[xr.DataArray]
    G: np.ndarray  # (kernel, vertex)
    sbf: dot.GaussianSpatialBasisFunctions
    ys: list[cdt.NDTimeSeries]
    c_meas: list[xr.DataArray]
    wavelengths: list[float]  # all wavelengths, sorted
    E: np.ndarray  # (wavelength, chromo)


def make_group(rng, channel, wavelength):
    """Sensitivity, time series and measurement variance of one group of channels."""
    values = rng.uniform(0.1, 20.0, size=(len(channel), N_BRAIN + N_SCALP, 2))
    values[:, N_BRAIN:, :] *= 5.0

    channel_coords = {
        "channel": channel,
        "source": ("channel", [f"S{c}" for c in channel]),
        "detector": ("channel", [f"D{c}" for c in channel]),
        "wavelength": wavelength,
    }

    Adot = xr.DataArray(
        values,
        dims=["channel", "vertex", "wavelength"],
        coords={
            "is_brain": ("vertex", [True] * N_BRAIN + [False] * N_SCALP),
            **channel_coords,
        },
    )

    y = xr.DataArray(
        rng.normal(size=(len(channel), 2, N_TIME)),
        dims=["channel", "wavelength", "time"],
        coords={**channel_coords, "time": np.arange(N_TIME, dtype=float)},
    )

    c_meas = xr.DataArray(
        rng.uniform(0.5, 2.0, size=(len(channel), 2)),
        dims=["channel", "wavelength"],
        coords={"channel": channel, "wavelength": wavelength},
    )

    return Adot, y, c_meas


@pytest.fixture
def mixed_case():
    rng = np.random.default_rng(seed=11)

    groups = [
        make_group(rng, channel, wavelength)
        for channel, wavelength in zip(MIXED_CHANNELS, MIXED_WAVELENGTHS)
    ]
    Adots, ys, c_meas = (list(g) for g in zip(*groups))

    G = np.zeros((N_KERNEL_BRAIN + N_KERNEL_SCALP, N_BRAIN + N_SCALP))
    G[:N_KERNEL_BRAIN, :N_BRAIN] = make_kernels(rng, N_KERNEL_BRAIN, N_BRAIN)
    G[N_KERNEL_BRAIN:, N_BRAIN:] = make_kernels(rng, N_KERNEL_SCALP, N_SCALP)

    wavelengths = sorted(set(np.concatenate(MIXED_WAVELENGTHS)))

    E = nirs.get_extinction_coefficients("prahl", wavelengths)
    E = E.pint.dequantify().sel(chromo=["HbO", "HbR"])
    E = E.transpose("wavelength", "chromo").values

    return MixedCase(Adots, G, make_sbf(Adots, G), ys, c_meas, wavelengths, E)


def reference_mixed_problems(case: MixedCase, recon_mode, use_sbf, brain_only):
    """Matrices and (wavelength, channel) row labels of each problem, and P."""
    groups = []
    for Adot in case.Adots:
        S, P = reference_basis(Adot, case.G, use_sbf, brain_only)
        groups.append((list(Adot.wavelength.values), list(Adot.channel.values), S))

    if recon_mode == "conc":
        blocks, rows = [], []
        for wavelengths, channels, S in groups:
            for i_wl, wl in enumerate(wavelengths):
                e = case.E[case.wavelengths.index(wl)]
                blocks.append(np.hstack([e[c] * S[:, :, i_wl] for c in range(len(e))]))
                rows.extend((wl, ch) for ch in channels)
        return [(np.vstack(blocks), rows)], P

    problems = []
    for wl in case.wavelengths:
        blocks, rows = [], []
        for wavelengths, channels, S in groups:
            if wl in wavelengths:
                blocks.append(S[:, :, wavelengths.index(wl)])
                rows.extend((wl, ch) for ch in channels)
        problems.append((np.vstack(blocks), rows))
    return problems, P


def values_at(arrays, rows):
    """Look up (wavelength, channel) rows in a list of arrays."""
    values = []
    for wl, ch in rows:
        array = next(
            a for a in arrays if ch in a.channel.values and wl in a.wavelength.values
        )
        values.append(array.sel(channel=ch, wavelength=wl).values)
    return np.stack(values)


@pytest.mark.parametrize("apply_c_meas", [False, True])
@pytest.mark.parametrize("alpha_spatial", [None, 1e-2])
@pytest.mark.parametrize("brain_only", [False, True])
@pytest.mark.parametrize("use_sbf", [False, True])
@pytest.mark.parametrize("recon_mode", ["conc", "mua"])
def test_mixed_probe_matches_dense_reference(
    mixed_case: MixedCase, recon_mode, use_sbf, brain_only, alpha_spatial, apply_c_meas
):
    case = mixed_case

    recon = dot.ImageRecon(
        case.Adots,
        recon_mode=recon_mode,
        brain_only=brain_only,
        spatial_basis_functions=case.sbf if use_sbf else None,
        alpha_meas=ALPHA_MEAS,
        alpha_spatial=alpha_spatial,
        apply_c_meas=apply_c_meas,
    )
    actual = recon.reconstruct(case.ys, case.c_meas if apply_c_meas else None)

    problems, P = reference_mixed_problems(case, recon_mode, use_sbf, brain_only)
    u = []
    for M, rows in problems:
        c = values_at(case.c_meas, rows) if apply_c_meas else None
        u.append(reference_W(M, alpha_spatial, c) @ values_at(case.ys, rows))
    expected = reference_image(u, P, recon_mode, case.E)

    if recon_mode == "mua":
        assert actual.wavelength.values.tolist() == case.wavelengths
    assert_matches(image_values(actual, recon_mode, ("vertex", "time")), expected)


@pytest.mark.parametrize("alpha_spatial", [None, 1e-2])
@pytest.mark.parametrize("brain_only", [False, True])
@pytest.mark.parametrize("recon_mode", ["conc", "mua"])
def test_mixed_probe_noise_matches_dense_reference(
    mixed_case: MixedCase, recon_mode, brain_only, alpha_spatial
):
    case = mixed_case

    recon = dot.ImageRecon(
        case.Adots,
        recon_mode=recon_mode,
        brain_only=brain_only,
        alpha_meas=ALPHA_MEAS,
        alpha_spatial=alpha_spatial,
        apply_c_meas=True,
    )
    actual = recon.get_image_noise(case.c_meas)

    problems, P = reference_mixed_problems(case, recon_mode, False, brain_only)
    cov = []
    for M, rows in problems:
        c = values_at(case.c_meas, rows)
        W = reference_W(M, alpha_spatial, c)
        cov.append(W @ np.diag(c) @ W.T)
    expected = reference_image_variance(cov, P, recon_mode, case.E)

    assert_matches(variance_values(actual, recon_mode), expected)


@pytest.mark.parametrize("recon_mode", ["conc", "mua"])
def test_mixed_probe_list_order_is_irrelevant(mixed_case: MixedCase, recon_mode):
    case = mixed_case
    kwargs = dict(recon_mode=recon_mode, alpha_meas=ALPHA_MEAS, apply_c_meas=True)

    expected = dot.ImageRecon(case.Adots, **kwargs).reconstruct(case.ys, case.c_meas)
    actual = dot.ImageRecon(case.Adots[::-1], **kwargs).reconstruct(
        case.ys[::-1], case.c_meas[::-1]
    )

    assert_matches(
        image_values(actual, recon_mode, ("vertex", "time")),
        image_values(expected, recon_mode, ("vertex", "time")),
    )


@pytest.mark.parametrize("use_sbf", [False, True])
@pytest.mark.parametrize("recon_mode", MODES)
def test_split_sensitivity_equals_whole(dense_case: DenseCase, recon_mode, use_sbf):
    """Splitting the channels of a probe into two groups changes nothing."""
    case = dense_case
    groups = [slice(0, 5), slice(5, None)]

    Adots = [case.Adot.isel(channel=g) for g in groups]
    ys = [case.y.isel(channel=g) for g in groups]
    c_meas = [case.c_meas.isel(channel=g) for g in groups]

    kwargs = dict(
        recon_mode=recon_mode,
        alpha_meas=ALPHA_MEAS,
        alpha_spatial=1e-2,
        apply_c_meas=True,
    )
    whole = dot.ImageRecon(
        case.Adot, spatial_basis_functions=case.sbf if use_sbf else None, **kwargs
    )
    split = dot.ImageRecon(
        Adots,
        spatial_basis_functions=make_sbf(Adots, case.G) if use_sbf else None,
        **kwargs,
    )

    assert_matches(
        image_values(split.reconstruct(ys, c_meas), recon_mode, ("vertex", "time")),
        image_values(whole.reconstruct(case.y, case.c_meas), recon_mode,
                     ("vertex", "time")),
    )
    assert_matches(
        variance_values(split.get_image_noise(c_meas), recon_mode),
        variance_values(whole.get_image_noise(case.c_meas), recon_mode),
    )
    assert_matches(
        variance_values(split.get_image_noise_posterior(c_meas), recon_mode),
        variance_values(whole.get_image_noise_posterior(case.c_meas), recon_mode),
    )


@pytest.mark.parametrize("recon_mode", MODES)
def test_single_element_list_equals_array(dense_case: DenseCase, recon_mode):
    case = dense_case
    kwargs = dict(
        recon_mode=recon_mode,
        alpha_meas=ALPHA_MEAS,
        alpha_spatial=1e-2,
        lambda_R_conc=1.0,
        apply_c_meas=True,
    )
    single = dot.ImageRecon(case.Adot, **kwargs)
    listed = dot.ImageRecon([case.Adot], **kwargs)

    xr.testing.assert_identical(
        listed.reconstruct([case.y], [case.c_meas]),
        single.reconstruct(case.y, case.c_meas),
    )
    xr.testing.assert_identical(
        listed.get_image_noise_posterior([case.c_meas]),
        single.get_image_noise_posterior(case.c_meas),
    )


def test_sbf_mask_covers_all_groups(mixed_case: MixedCase):
    case = mixed_case
    sbf = dot.GaussianSpatialBasisFunctions.__new__(dot.GaussianSpatialBasisFunctions)
    sbf.mask_threshold = 1.5

    # each group contributes the sensitivity at its first wavelength
    intensity = sum(A.isel(wavelength=0).sum("channel") for A in case.Adots)
    expected = np.log10(intensity.values) > sbf.mask_threshold

    sbf._compute_sensitivity_mask(case.Adots)

    np.testing.assert_array_equal(sbf._mask.values, expected)
    assert isinstance(case.sbf.H, list) and len(case.sbf.H) == 2


def test_mixed_probe_rejects_mua2conc(mixed_case: MixedCase):
    with pytest.raises(ValueError, match="mua2conc"):
        dot.ImageRecon(mixed_case.Adots, recon_mode="mua2conc")


def test_mixed_probe_rejects_different_vertices(mixed_case: MixedCase):
    Adots = mixed_case.Adots
    other = Adots[1].assign_coords(is_brain=("vertex", ~Adots[1].is_brain.values))

    with pytest.raises(ValueError, match="same 'vertex' coords"):
        dot.ImageRecon([Adots[0], other])

    with pytest.raises(ValueError, match="same 'vertex' coords"):
        dot.ImageRecon([Adots[0], Adots[1].isel(vertex=slice(1, None))])


def test_mixed_probe_rejects_duplicate_channels(mixed_case: MixedCase):
    with pytest.raises(ValueError, match="only one sensitivity"):
        dot.ImageRecon([mixed_case.Adots[0], mixed_case.Adots[0]])


def test_mixed_probe_rejects_sbf_of_other_sensitivities(mixed_case: MixedCase):
    sbf = make_sbf(mixed_case.Adots[0], mixed_case.G)

    with pytest.raises(ValueError, match="different sensitivities"):
        dot.ImageRecon(mixed_case.Adots, spatial_basis_functions=sbf)


def test_y_with_unknown_channels_is_rejected(mixed_case: MixedCase):
    recon = dot.ImageRecon(mixed_case.Adots[:1], alpha_meas=ALPHA_MEAS)

    with pytest.raises(ValueError, match="not in the sensitivity matrix"):
        recon.reconstruct(mixed_case.ys)


def test_y_with_different_time_axes_is_rejected(mixed_case: MixedCase):
    recon = dot.ImageRecon(mixed_case.Adots, alpha_meas=ALPHA_MEAS)
    shifted_time = np.arange(N_TIME) + 0.5
    ys = [mixed_case.ys[0], mixed_case.ys[1].assign_coords(time=shifted_time)]

    with pytest.raises(ValueError):
        recon.reconstruct(ys)


def test_mixed_probe_posterior_with_lambda_R_conc_is_not_implemented(
    mixed_case: MixedCase,
):
    recon = dot.ImageRecon(
        mixed_case.Adots, recon_mode="mua", alpha_meas=ALPHA_MEAS, lambda_R_conc=1.0
    )

    with pytest.raises(NotImplementedError):
        recon.get_image_noise_posterior()
