import numpy as np
from scipy.signal import butter, filtfilt



def TDDR(data, Fs, split_PosNeg, usePCA):

    if(hasattr(data,'wavelength')):
        data=data.transpose('time','channel','wavelength')
    else:
        data=data.transpose('time','channel','chromo')

    units=data.pint.units
    #d.pint.dequalify()
    shp=data.shape
    d=np.reshape(data.as_numpy().data,(shp[0],shp[1]*shp[2]))

    if usePCA:
        U, S, V = np.linalg.svd(d)
        U=U[:,:len(S)]
        S=np.diag(S)
        U = local_tddr(U, Fs, split_PosNeg)

        d = U @ S @ V.T
    else:
        d = local_tddr(d, Fs, split_PosNeg)

    data.data=np.reshape(d,shp)
    return data

def local_tddr(signal, sample_rate, splitPosNeg=False):
    """
    Perform Temporal Derivative Distribution Repair (TDDR) motion correction

        Usage:
        signals_corrected = nirs.math.tddr( signals , sample_rate );

        Inputs:
        signals: A [sample x channel] matrix of uncorrected optical density data
        sample_rate: A scalar reflecting the rate of acquisition in Hz

        Outputs:
        signals_corrected: A [sample x channel] matrix of corrected optical density data

        Fishburn F.A., Ludlum R.S., Vaidya C.J., & Medvedev A.V. (2019). 
        Temporal Derivative Distribution Repair (TDDR): A motion correction 
        method for fNIRS. NeuroImage, 184, 171-179.
        https://doi.org/10.1016/j.neuroimage.2018.09.025
    """
    
    # Iterate over each channel
    nch = signal.shape[1] if signal.ndim > 1 else 1
    if nch > 1:
        signal_corrected = np.zeros_like(signal)
        for ch in range(nch):
            signal_corrected[:, ch] = local_tddr(signal[:, ch], sample_rate, splitPosNeg)
        return signal_corrected

    DC = np.median(signal)
    signal = signal - DC

    # Preprocess: Separate high and low frequencies
    filter_cutoff = 0.5
    filter_order = 3
    Fc = filter_cutoff * 2 / sample_rate
    if Fc < 1:
        fb, fa = butter(filter_order, Fc)
        signal_low = filtfilt(fb, fa, signal)
    else:
        signal_low = signal
    signal_high = signal - signal_low

    # Initialize
    tune = 4.685
    D = np.sqrt(np.finfo(signal.dtype).eps)
    mu = np.inf
    iter = 0

    # Step 1. Compute temporal derivative of the signal
    deriv = np.diff(signal_low)

    # Step 2. Initialize observation weights
    w = np.ones_like(deriv)

    # Step 3. Iterative estimation of robust weights
    while iter < 50:
        iter += 1
        mu0 = mu

        # Step 3a. Estimate weighted mean
        mu = np.sum(w * deriv) / np.sum(w)

        if splitPosNeg:
            lst = np.where((deriv - mu) > 0)[0]
            if(len(lst)>0):
                # Step 3b. Calculate absolute residuals of estimate
                dev = np.abs(deriv[lst] - mu)

                # Step 3c. Robust estimate of standard deviation of the residuals
                sigma = 1.4826 * np.median(dev)

                # Step 3d. Scale deviations by standard deviation and tuning parameter
                r = dev / (sigma * tune)

                # Step 3e. Calculate new weights according to Tukey's biweight function
                w[lst] = ((1 - r**2) * (r < 1)) ** 2

                lst = np.where((deriv - mu) <= 0)[0]
                # Step 3b. Calculate absolute residuals of estimate
                dev = np.abs(deriv[lst] - mu)

                # Step 3c. Robust estimate of standard deviation of the residuals
                sigma = 1.4826 * np.median(dev)

                # Step 3d. Scale deviations by standard deviation and tuning parameter
                r = dev / (sigma * tune)

                # Step 3e. Calculate new weights according to Tukey's biweight function
                w[lst] = ((1 - r**2) * (r < 1)) ** 2
        else:
            # Step 3b. Calculate absolute residuals of estimate
            dev = np.abs(deriv - mu)

            # Step 3c. Robust estimate of standard deviation of the residuals
            sigma = 1.4826 * np.median(dev)

            # Step 3d. Scale deviations by standard deviation and tuning parameter
            r = dev / (sigma * tune)

            # Step 3e. Calculate new weights according to Tukey's biweight function
            w = ((1 - r**2) * (r < 1)) ** 2

        # Step 3f. Terminate if new estimate is within machine-precision of old estimate
        if np.abs(mu - mu0) < D * np.max([np.abs(mu), np.abs(mu0)]):
            break

    # Step 4. Apply robust weights to centered derivative
    new_deriv = w * (deriv - mu)

    # Step 5. Integrate corrected derivative
    signal_low_corrected = np.cumsum(np.concatenate(([0], new_deriv)))

    # Postprocess: Center the corrected signal
    signal_low_corrected = signal_low_corrected - np.mean(signal_low_corrected)

    # Postprocess: Merge back with uncorrected high frequency component
    signal_corrected = signal_low_corrected + signal_high

    signal_corrected = signal_corrected + DC

    return signal_corrected

