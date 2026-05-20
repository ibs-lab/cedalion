import numpy as np

# import cedalion.dataclasses.statistics
import statsmodels.api as sm
import cedalion.math.ar_model
import scipy.signal
import pandas as pd
from cedalion.sigproc.frequency import sampling_rate
from cedalion import cite, units

def ar_irls_GLM(y, x, pmax: int | None = 40, M=sm.robust.norms.TukeyBiweight(c=4.685)):
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
    mask = np.isfinite(y.values)

    if pmax is None:
        fs = sampling_rate(y).to(units.Hz)
        pmax = 2 * np.ceil(fs)

    yorg : pd.Series = pd.Series(y.values[mask].copy())
    xorg : pd.DataFrame = x[mask].reset_index(drop=True)

    y = yorg.copy()
    x = xorg.copy()

    rlm_model = sm.RLM(y, x, M=M)
    params = rlm_model.fit()

    resid = pd.Series(y - x @ params.params)
    for _ in range(4):  # TODO - check convergence
        y = yorg.copy()
        x = xorg.copy()

        # Update the AR whitening filter
        arcoef = cedalion.math.ar_model.bic_arfit(resid, pmax=pmax)
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
        # yet fully initialized.
        rlm_model = sm.RLM(yf[p:], xf.iloc[p:], M=M)
        params = rlm_model.fit()

        resid = pd.Series(yorg - xorg @ params.params)

    return params
