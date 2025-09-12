import numpy as np
import statsmodels.api as sm
import pandas as pd
from scipy.signal import lfilter


def bic_arfit(dd, pmax=30):
    """This function computes the ar coefficients up to a max model order.

    BIC is used to select the model

    Args:
        dd: pd.Series
        pmax: int (default 30)

    Returns:
        sm.tsa.AutoReg results class (includes intercept term)
    """

    dd = dd - dd.mean()

    last_bic = np.inf
    for p in range(pmax + 1):
        model = sm.tsa.AutoReg(dd, lags=p).fit()
        if model.bic > last_bic:
            break
        else:
            last_bic = model.bic

    # Select the best model order
    best_p = p - 1
    return sm.tsa.AutoReg(dd, lags=best_p).fit()


def fit_ar_coefs(data, pmax=12):
    """This function loops over a timeseries and computs the AR-coefficients.

    Args:
        data: xr.DataArray time course
        pmax: int (default 12)

    Returns:
        Array[channels,types] of lists of AR stats results classes
    """
    nchan = len(data.channel)

    if hasattr(data, "wavelength"):
        ntype = len(data.wavelength)
        data = data.transpose("time", "channel", "wavelength")
    else:
        ntype = len(data.chromo)
        data = data.transpose("time", "channel", "chromo")

    # Initialize the array of list
    ar_coeffs = [[[] for _ in range(ntype)] for _ in range(nchan)]

    for chIdx in range(nchan):
        for wIdx in range(ntype):
            dd = pd.Series(data[:, chIdx, wIdx].to_numpy())
            model = bic_arfit(dd, pmax=pmax)
            # Get the AR model coefficients
            ar_coeffs[chIdx][wIdx].append(model)

    return ar_coeffs


def ar_filter(data, pmax=12):
    """This function computes and applies an AR filter on a data time series.

    Args:
        data: xr.DataArray
        pmax: int (default=12)
    """

    if hasattr(data, "wavelength"):
        data = data.transpose("time", "channel", "wavelength")
    else:
        data = data.transpose("time", "channel", "chromo")

    data2 = data.copy()
    armodels = fit_ar_coefs(data2, pmax)

    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            # grab the ar coeffieints (ignoring the DC term)
            params = armodels[i][j][0].params[1:]

            # Convert to an IIR filter design
            wf = np.hstack([1, -params])
            dd = data[:, i, j] - data[:, i, j].mean()
            # Apply the filter
            data2.pint.magnitude[:, i, j] = lfilter(wf, [1], dd.to_numpy())

    return data2
