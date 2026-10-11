import numpy as np

# import cedalion.dataclasses.statistics
import statsmodels.api as sm
import cedalion.math.ar_model
import scipy.signal
import pandas as pd
from cedalion.sigproc.frequency import sampling_rate
from cedalion import cite, units

def _valid_after_filter(mask: np.ndarray, p: int) -> np.ndarray:
    """Find the samples of an AR-filtered series that only depend on unmasked samples.

    The output of an AR whitening filter of order p at sample t depends on the input
    samples t-p, ..., t. It is valid only if all of these are unmasked. The first p
    samples, for which the filter is not yet initialized, are never valid.

    Args:
        mask: 1-D boolean array, ``True`` for unmasked samples.
        p: order of the AR filter.

    Returns:
        1-D boolean array, ``True`` for valid samples of the filtered series.
    """
    masked = (~mask).astype(int)
    invalid = np.convolve(masked, np.ones(p + 1, dtype=int))[: len(mask)] > 0
    invalid[:p] = True
    return ~invalid


def ar_irls_GLM(
    y,
    x,
    pmax: int | None = 40,
    M=sm.robust.norms.TukeyBiweight(c=4.685),
    mask: np.ndarray | None = None,
):
    """This function implements the AR-IRLS GLM model.

    The autoregressive iteratively reweighted least squares GLM model is described in
    :cite:t:`Barker2013`. By estimating prewhitening filters it addresses serial
    correlations and confounding noise components in the signal and avoids the inflated
    false positive rates observed when fitting the GLM with ordinary least squares.

    Args:
        y: Dependent variable time series (pandas Series or NDTimeSeries with a
            ``"time"`` dimension).
        x: Design matrix (pandas DataFrame).
        pmax: Maximum AR model order to consider. If ``None``, set to
            ``2 × ceil(sampling_rate)``. A value of 4–5× the sampling rate is
            sufficient; do not set it excessively high as it reduces the number
            of usable time points.
        M: Robust norm for the IRLS step (default: Tukey bi-weight, ``c=4.685``).
            Tuning constant reference values: 4.685 → 95% efficiency,
            4.00 → ~90%, 3.55 → ~85%.
        mask: Optional 1-D boolean array. Samples that are ``False`` are ignored in
            the fit. Filtered samples within the AR order after a masked sample are
            ignored, too, because the whitening filter would mix in masked samples.
            If ``None``, all non-finite samples of ``y`` are masked.

    Returns:
        Fitted ``statsmodels.RLM`` results object containing betas, t-stats,
        and residuals after prewhitening.

    Note:
        Do **not** low-pass filter before calling this function — the AR step
        needs access to the full spectrum to estimate stable prewhitening filters.
        High-pass filtering is acceptable; prefer Legendre polynomials or
        discrete cosine terms in the design matrix over regular polynomials.

    Initial Contributors:
        Ted Huppert | huppert1@pitt.edu | 2024
    """

    cite("Barker2013")
    if mask is None:
        mask = np.isfinite(y.values)
    else:
        mask = np.asarray(mask, dtype=bool)

    if pmax is None:
        fs = sampling_rate(y).to(units.Hz)
        pmax = 2 * np.ceil(fs)

    # Keep all samples to preserve their temporal order for the AR filter. The values
    # of masked samples are replaced but never enter the fit.
    yorg : pd.Series = pd.Series(np.where(mask, y.values, 0.0))
    xorg : pd.DataFrame = x.reset_index(drop=True)

    y = yorg.copy()
    x = xorg.copy()

    rlm_model = sm.RLM(y[mask], x[mask], M=M)
    params = rlm_model.fit()

    resid = pd.Series(y - x @ params.params)
    for _ in range(4):  # TODO - check convergence
        y = yorg.copy()
        x = xorg.copy()

        # Update the AR whitening filter. The AR coefficients are estimated from the
        # concatenated residuals of the unmasked samples.
        arcoef = cedalion.math.ar_model.bic_arfit(
            resid[mask].reset_index(drop=True), pmax=pmax
        )
        wf = np.hstack([1, -arcoef.params[1:]])
        p = len(wf) - 1

        # Apply the AR filter to the lhs and rhs of the model
        yf = pd.Series(scipy.signal.lfilter(wf, 1, y))

        xf = np.zeros(x.shape)
        xx = x.to_numpy()
        for i in range(xx.shape[1]):
            xf[:, i] = scipy.signal.lfilter(wf, 1, xx[:, i])

        xf = pd.DataFrame(xf)
        xf.columns = x.columns

        # fit the model ignoring the first p samples, for which the AR filter is not
        # yet fully initialized, and the filtered samples affected by masked samples.
        valid = _valid_after_filter(mask, p)
        rlm_model = sm.RLM(yf[valid], xf[valid], M=M)
        params = rlm_model.fit()

        resid = pd.Series(yorg - xorg @ params.params)

    return params
