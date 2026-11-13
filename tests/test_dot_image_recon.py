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


def reference_basis(case: DenseCase, use_sbf, brain_only):
    """Sensitivity S (channel, unknown, wavelength) and P (vertex, unknown)."""
    n_vertex = N_BRAIN + N_SCALP
    S = case.Adot.values
    P = np.eye(n_vertex)
    unknown_is_brain = np.arange(n_vertex) < N_BRAIN

    if use_sbf:
        S = np.einsum("cvw,kv->ckw", S, case.G)
        P = case.G.T
        unknown_is_brain = np.arange(case.G.shape[0]) < N_KERNEL_BRAIN

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

    S, P = reference_basis(case, use_sbf, brain_only)
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

    S, P = reference_basis(case, use_sbf, brain_only)
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

    S, P = reference_basis(case, use_sbf, brain_only)
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
    S, P = reference_basis(case, False, False)
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
