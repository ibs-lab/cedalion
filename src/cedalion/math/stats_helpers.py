import numpy as np


def BenjaminiHochberg(p : np.ndarray):
    """Apply Benjamini-Hochberg FDR correction for multiple comparisons.

    See http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3530395/

    Args:
        p: Array of uncorrected p-values, arbitrary shape.

    Returns:
        Array of FDR-corrected q-values, same shape as ``p``, clipped to [0, 1].
    """

    s = p.shape

    # sort the p values
    p, II = np.sort(p.flatten()), np.argsort(p.flatten())

    # number of hypothesis tests
    m = len(p)

    if m < 100:
        # Benjamini-Hochberg Procedure
        pi0 = 1

    elif m < 1000:
        # Storey's method w/ point estimate
        pi0 = 0.5 / (1 - p[m // 2])

    else:
        # Storey's method w/ interpolation
        # grid to estimate fraction of null tests
        x = p
        y = (m - np.arange(1, m + 1)) / m / (1 - x)

        # only in this range
        lst = (x > 0.1) & (x < 0.85)
        y = y[lst]
        x = x[lst]

        # estimate
        X = np.column_stack((np.ones(x.shape), x, x**2, x**3))
        b = np.linalg.lstsq(X, y, rcond=None)[0]

        # interpolate
        pi0 = np.dot(np.array([1, 1, 1, 1]), b)

        pi0 = max(pi0, 0)
        pi0 = min(pi0, 1)

    pi0 = 1 # disable adaptive FDR as it is done in Brain AnalyzIR

    # p = (i/m)*Q
    q = p * m / np.arange(1, m + 1) * pi0

    for i in range(len(q) - 1, 0, -1):
        if q[i] < q[i - 1]:
            q[i - 1] = q[i]

    # put them back in the original order
    q[II] = q

    q = q.reshape(s)

    q[q > 1] = 1

    return q
