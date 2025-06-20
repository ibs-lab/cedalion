import numpy as np

# import cedalion.dataclasses.statistics
import statsmodels.api as sm
import cedalion.math.ar_model
import scipy.signal
import pandas as pd


def ar_irls_GLM(y, x, pmax=40, M=sm.robust.norms.HuberT()):
    """This function implements the AR-IRLS GLM model.

    The autoregressive iteratively reweighted least squares GLM model is described in
    :cite:t:`Barker2013`. By estimating prewhitening filters it addresses serial
    correlations and confounding noise components in the signal and avoids the inflated
    false positive rates observed when fitting the GLM with ordinary least squares.

    Inputs:
        y - pandas Serial
        x - pandas DataFrame
        pmax- max AR model order (default 40)
        M- statsmodel.robust.norms type (default Huber)

    Outputs:
        stats- statsmodel.RLM results model

    Initial Contributors:
        Ted Huppert | huppert1@pitt.edu | 2024

      d is matrix containing the data; each column is a channel of data

      X is the regression/design matrix

      Pmax is the maximum AR model order that you want to consider. A
      purely speculative guess is that the average model order is
      approximatley equal to the 2-3 times the sampling rate, so setting Pmax
      to 4 or 5 times the sampling rate should work fine.  The code does not
      suffer a hugeperformance hit by using a higher Pmax; however, the number
      of time points used to estimate the AR model will be
      "# of time points - Pmax", so don't set Pmax too high.

      "tune" is the tuning constant used for Tukey's bisquare function during
      iterative reweighted least squares. The default value is 4.685.
      Decreasing "tune" will make the regression less sensitive to outliers,
      but at the expense of performance (statistical efficiency) when data
      does not have outliers. For reference, the values of tune for 85, 90,
      and 95  statistical efficiency are

      tune = 4.685 --> 95%
      tune = 4.00  --> ~90%
      tune = 3.55  --> ~85%

      I have not tested these to find an optimal value for the "average" NIRS
      dataset; however, 4.685 was used in the published simulations and worked
      quite well even with a high degree of motion artifacts from children.
      If you really want to adjust it, you could use the above values as a
      guideline.

      DO NOT preprocess your data with a low pass filter.
      The algorithm is trying to transform the residual to create a
      white spectrum.  If part of the spectrum is missing due to low pass
      filtering, the AR coefficients will be unstable.  High pass filtering
      may be ok, but I suggest putting orthogonal polynomials (e.g. Legendre)
      or low frequency discrete cosine terms directly into the design matrix
      (e.g. from spm_dctmtx.m from SPM).  Don't use regular polynomials
      (e.g. 1 t t^2 t^3 t^4 ...) as this can result in a poorly conditioned
      design matrix.

      If you choose to resample your data to a lower sampling frequency,
      makes sure to choose an appropriate cutoff frequency so that that the
      resulting time series is not missing part of the frequency spectrum
      (up to the Nyquist bandwidth).  The code should work fine on 10-30 Hz
      data.
    """

    mask = np.isfinite(y.values)

    yorg : pd.Series = pd.Series(y.values[mask].copy())
    xorg : pd.DataFrame = x[mask].reset_index()

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

        # Apply the AR filter to the lhs and rhs of the model
        a = y[0]
        yf = pd.Series(scipy.signal.lfilter(wf, 1, y - a)) + a
        xf = np.zeros(x.shape)
        xx = x.to_numpy()
        for i in range(xx.shape[1]):
            b = xx[0, i]
            xf[:, i] = scipy.signal.lfilter(wf, 1, xx[:, i] - b) + b

        xf = pd.DataFrame(xf)
        xf.columns = x.columns

        rlm_model = sm.RLM(yf, xf, M=M)
        params = rlm_model.fit()

        resid = pd.Series(yorg - xorg @ params.params)

    return params
