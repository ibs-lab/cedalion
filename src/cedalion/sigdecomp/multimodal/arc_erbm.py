"""Independent Component Analysis by Entropy Bound Rate Minimization (ICA-ERBM).

This code is based on :cite:t:`Li2010B` and :cite:t:`Fu2014`. It was converted from
matlab versions provided by the MLSP Lab at the University of Maryland, which is
available here: https://mlsp.umbc.edu/resources.html.
"""


import scipy as sp
import numpy as np
from cedalion.sigdecomp.multimodal import arc_ebm as arc_ebm
import cedalion.data

def arc_erbm(X: np.ndarray, guess_mat, p: int = None , pr_guess_mat = None) -> np.ndarray:
    """ Adaptive-reverse Constrained ICA by Entropy Rate Bound Minimization (arc-ERBM) is a spectrally constrained ICA algorithm. 

    Args:
        X (np.ndarray, (Channels, Time Points)): the [N x T] input multivariate time series with dimensionality N observations/channels and T time points

        guess_mat (np.ndarray, (Time Points/2 , Referenced Channels)): Frequency reference signal for the reconstruction

        p (int): the filter length for the invertible filter source model, does not need to be specified. Default is p = minimum(11, T/50).

        pr_guess_mat (np.ndarray, (Time Points, Referenced Channels)):  Optional time domain reference signal for the reconstruction, however, only frequency characteristics are used. Only needed if Phase Retrieval Projection constraint should be included.

    Returns:
        W (np.ndarray, (Channels, Channels)): the [N x N] demixing matrix with weights for N channels/sources. To obtain the independent components,
        the demixed signals can be calculated as S = W @ X.

    Initial Contributors:
        - Jacqueline Behrendt | j.behrendt@tu-berlin.de | 2026

    References:
        This code is based on the matlab version of bss by Xi-Lin Li (:cite:t:`Li2010B`)
        Xi-Lin Li, Tulay Adali, "Blind spatiotemporal separation of second and/or
        higher-order correlated sources by entropy rate minimization,"
        IEEE International Conference on Acoustics, Speech and Signal Processing 2010.
        The original matlab version is available at https://mlsp.umbc.edu/resources.html
        under the name "Real-valued ICA by entropy bound minimization (ICA-EBM)"
    """

#################  Part 0: pre-processing #################

    # load measuring functions as global variables
    global nf1, nf3, nf5, nf7

    file_path = cedalion.data.get("measfunc_table.npy")
    table = np.load(file_path, allow_pickle=True)

    K = 8
    nf1, nf3, nf5, nf7 = table[0], table[2], table[4], table[6] 
   
    # Apply pre-processing to data
    N, T = X.shape
    X, P = pre_processing(X)

    # initialize p if it is not provided
    if p is None:
        p = int(np.minimum(11, T/ 50))

    if pr_guess_mat is  None:
        constraint = 'psd'
    else: 
        constraint = 'phase_retrieval'

    # Define similarity measures and gradients for constraints
    # normalize reference signals
    for i in range(guess_mat.shape[1]): 
        r_n_c = guess_mat[:, i]   
        r_n_c = r_n_c - np.mean(r_n_c) 
        guess_mat[:, i] = np.copy(r_n_c / np.linalg.norm(r_n_c) )

    # compute psd of X 
    X_hat = (2/T) * np.fft.rfft(X, axis = 1) 

    # compute cross psd of X 
    C_hat = np.zeros((X_hat.shape[1], N , N ), dtype=complex)
    C_hat = (X_hat[:, None, :] * np.conjugate(X_hat[None, :, :])).transpose(2, 0, 1)
    
    # center C_hat  
    C_hat_mean = np.mean(C_hat, axis = 0)   
    C_hat = C_hat - np.reshape(C_hat_mean, (1, N, N))   

    # store real matrix for gradient computation 
    C_tilde = np.real(C_hat + np.transpose(C_hat, (0, 2, 1)) )
    
    # define similarity measure for psd constraint
    def epsilon(a,b): 
        # power spectral density 
        # b is already a psd 
        psd_a = (2/T) * np.abs(np.fft.rfft(a))**2  
        psd_correlation = np.corrcoef(psd_a, b)[0,1] 
        return psd_correlation 
    
    def epsilon_grad(r_n_c): 
        # compute gradient of epsilon for the psd constraint   
        # compute psd of estimated source
        psd_s = (2/T) * np.abs(np.fft.rfft(w.T.dot(X)))**2 

        # compute correlation between estimated and reference psd
        current_corr = np.corrcoef(psd_s, r_n_c)[0,1]
        sign = np.sign(current_corr) 

        # compute gradient vector
        vec = (sign / np.linalg.norm(psd_s, 2)) * r_n_c -  (np.abs(current_corr) / np.linalg.norm(psd_s, 2)**2) * psd_s 
        vec = vec.reshape((-1, 1, 1))     
        c_grad  = (T/2) * mu_c[n] * np.sum( np.multiply(np.dot(C_tilde,  w), vec), axis = 0  )    
        return c_grad
             
    if constraint == 'phase_retrieval' : 
        amplitude = pr_guess_mat

        def pr_update(amp, y, filter, X_filtered): 
            # this function applies a phase retrieval update step

            # filter reference amplitude
            amp_filtered =  sp.signal.lfilter(filter, 1, amp , axis = 0 )
            amp_filtered = np.abs(np.fft.rfft(amp_filtered)) 

            # compute fft of estimated source
            y_hat = np.fft.rfft(y)

            # project onto magnitude constraint
            g_hat = amp_filtered * np.exp(1j * np.angle(y_hat))
            g_hat = g_hat.reshape((-1, 1))

            # inverse fft to time domain
            g = np.fft.irfft(g_hat, axis = 0 , n =X_filtered.shape[1] )

            # compute corresponding weights 
            returns = np.linalg.lstsq(X_filtered.T, g.flatten())
            v_tilde = returns[0]

            return v_tilde

            
    # initialize W
    W = arc_ebm.arc_ebm(X, guess_mat, 'psd')
    
    gam = 2500
    decaying_factor = 0.99 
        
    # Choose set of threshold values
    rho = np.arange(0, 1.01, 0.01) 

    
    # Number of reference signals   
    num_guess = guess_mat.shape[1] 

    mu_c = np.ones((num_guess, 1))  
    corr_w_guess = np.zeros((num_guess, N)) 
    num_W = np.shape(W)[0]  
    corr_w_guess = np.zeros((num_guess, num_W)) 

    # Resort W based on correlation with reference signals 
    for kl in range(num_guess): 
        r_n_c = guess_mat[:, kl]    
        for lp in range(num_W): 
            w = W[lp, :].T
            corr_w_guess[kl, lp] = epsilon(X.T.dot(w), r_n_c)  
            
    # May need auction to auction to choose order 
    max_index = np.argmax(np.abs(corr_w_guess), axis = 1)  
    if len(np.unique(max_index)) != num_guess: 
        colsol, _ = auction((1- np.abs(corr_w_guess)).T)
        max_index = colsol.T
    c = np.arange(0, num_W) 
    c = np.setdiff1d(c, max_index)  
    sort_order = np.concatenate((max_index, c)) 
    W = W[sort_order, :] 


    if p == 1:
        W = W.dot(P)
        return W

    # prediction coefficients
    a = np.zeros((p, N ))
    for n in range(N):
       a[int(np.floor((p+1)/2) - 1 ), n ] = 1

    Rz = np.zeros((N, N, N))
    temp5 = np.zeros((T, N))
    Z = np.zeros((N,T,N))

    # Prepare the data used in integral
    calculate_cos_sin_mtx(p)

        
    last_W = np.copy(W)
    best_W = np.copy(W)
    best_a = np.copy(a) 

    #################  Part 1: #################
    for stochastic_search in range(1,-1, -1):
        if stochastic_search ==1 :
            mu = 1/5
            max_cost_increase = 5
            max_iter_north = 500
            tolerance = 1e-3
        else:
            mu = 1/50
            max_cost_increase = 3
            max_iter_north = 200
            tolerance = 1e-5

        cost_increase_counter = 0   
        mu_idx = np.full(mu_c.shape, False)

        min_mu = 1/200  
        mu_old = np.copy(mu_c)
        rho_n_arr = np.zeros((max_iter_north+1, N))  

        W = np.copy(best_W)
        a = np.copy(best_a)
        last_W = np.copy(best_W)
        Cost = np.zeros((max_iter_north+1, 1))
        min_cost = np.inf
        min_cost_queue = min_cost * np.ones((max_iter_north+1, 1))
        negentropy_array = np.zeros((N,1))
            
        for iter in range(1, max_iter_north+1):

            if stochastic_search == 1:
                # estimate AR coefficients
                Y = np.copy(np.dot(W, X) )
                for n in range(N):

                    if iter%6 == 1 or iter<= 5:
                        
                        a1, min_ere1  = lfc(Y[n,:], p , 'unknown', [])
                        a2, min_ere2 = lfc(Y[n, :], p, [], a[:, n])

                        # choose the best model
                        min_ere = np.inf
                        if min_ere > min_ere1:
                            min_ere = min_ere1
                            a[:, n] = np.copy(a1)
                        if min_ere > min_ere2:
                            min_ere = min_ere2
                            a[:, n] = np.copy(a2)

                    elif iter%6 == 4 :
                        a3, _ = lfc(Y[n, :], p, [], a[:, n])
                        a[:, n ] = np.copy(a3)

                    temp5 =  sp.signal.lfilter(a[:, n].T, 1, X.T , axis = 0 )
                    Rz[ :, :, n] = np.dot(temp5.T, temp5) / T
                    Z[:, :, n] = np.copy(temp5.T)

            Cost[iter-1] = np.copy(- np.log(np.abs(np.linalg.det(W))))

            # estimate W
            for n in range(N):
                temp1 = np.random.rand(N, 1)
                temp2 = np.delete(W, n, axis = 0)
                h = temp1 - temp2.T.dot( np.linalg.solve( np.dot(temp2, temp2.T), temp2)).dot(temp1 )
                v = np.copy(W[n, :].T )
                sigma2 = v.T.dot(Rz[:, :, n]).dot(v)
                Cost[iter-1] = np.copy(Cost[iter-1] + np.log(sigma2)/2 )
                v = np.copy(v / np.sqrt(sigma2))

                # prediction error
                y = np.copy(v.T.dot(Z[:, :, n   ]))

                # evaluate the upper bound of negentropy of the n-th component
                NE_Bound = np.zeros((K, 1))
                EGx = np.zeros((K, 1))
                # we only need to calculate these quantities once
                yy = y* y
                sign_y = np.sign(y)
                abs_y = np.abs(y)
                inv_pabs_y = 1/(1 + abs_y)
                inv_pabs_yy = 1/(1+ yy)
                inv_p10abs_y = 1/(10+abs_y)

                # G1(x) = x^4
                EGx[0] = np.sum(yy*yy)/T
                if EGx[0] < nf1['min_EGx']:
                    NE_Bound[0] = simplified_ppval(nf1['pp_slope'], nf1['min_EGx'] ) * (EGx[0] - nf1['min_EGx'])
                    NE_Bound[0] = simplified_ppval(nf1['pp'],nf1['min_EGx'])  + np.abs(NE_Bound[0] )
                else:
                    if EGx[0] > nf1['max_EGx']:
                        NE_Bound[0] = 0
                    else:
                        NE_Bound[0] = simplified_ppval(nf1['pp'], EGx[0] )

                # G3(x) = np.abs(x)/ (1 + np.abs(x))
                EGx[2] = 1 - np.sum(inv_pabs_y)/T
                if EGx[2] < nf3['min_EGx']:
                    NE_Bound[2] = simplified_ppval(nf3['pp_slope'], nf3['min_EGx'] ) * (EGx[2] - nf3['min_EGx'])
                    NE_Bound[2] = simplified_ppval(nf3['pp'], nf3['min_EGx']) + np.abs(NE_Bound[2])
                else:
                    if EGx[2] > nf3['max_EGx']:
                        NE_Bound[2] = simplified_ppval(nf3['pp_slope'], nf3['max_EGx'] ) * (EGx[2] - nf3['max_EGx'])
                        NE_Bound[2] = simplified_ppval(nf3['pp'], nf3['max_EGx']) + np.abs(NE_Bound[2])

                    else:
                        NE_Bound[2] = simplified_ppval(nf3['pp'], EGx[2] )

                # G5(x)  = x* np.abs(x) /(10 + np.abs(x))
                EGx[4] = np.sum( y * abs_y * inv_p10abs_y )/T
                if EGx[4] < nf5['min_EGx']:
                    NE_Bound[4] = simplified_ppval(nf5['pp_slope'], nf5['min_EGx'] ) * (EGx[4] - nf5['min_EGx'])
                    NE_Bound[4] = simplified_ppval(nf5['pp'], nf5['min_EGx']) + np.abs(NE_Bound[4])
                else:
                    if EGx[4] > nf5['max_EGx']:
                        NE_Bound[4] = simplified_ppval(nf5['pp_slope'], nf5['max_EGx'] ) * (EGx[4] - nf5['max_EGx'])
                        NE_Bound[4] = simplified_ppval(nf5['pp'], nf5['max_EGx']) + np.abs(NE_Bound[4])
                    else:
                        NE_Bound[4] = simplified_ppval(nf5['pp'], EGx[4] )

                # G7(x) =  x / (1 + x**2)
                EGx[6] = np.sum(y*inv_pabs_yy)/T
                if EGx[6] < nf7['min_EGx']:
                    NE_Bound[6] = simplified_ppval(nf7['pp_slope'], nf7['min_EGx'] ) * (EGx[6] - nf7['min_EGx'])
                    NE_Bound[6] = simplified_ppval(nf7['pp'], nf7['min_EGx']) + np.abs(NE_Bound[6])
                else:
                    if EGx[6] > nf7['max_EGx']:
                        NE_Bound[6] = simplified_ppval(nf7['pp_slope'], nf7['max_EGx'] ) * (EGx[6] - nf7['max_EGx'])
                        NE_Bound[6] = simplified_ppval(nf7['pp'], nf7['max_EGx']) + np.abs(NE_Bound[6])
                    else:
                        NE_Bound[6] = simplified_ppval(nf7['pp'], EGx[6] )

                # select the tightest upper bound
                max_NE, max_i = np.max(NE_Bound), np.argmax(NE_Bound)
                negentropy_array[n] = np.copy(max_NE)
                Cost[iter -1] = np.copy(Cost[iter-1] - max_NE)

                if n < num_guess: 
                    # choose reference signal
                    r_n_c = guess_mat[:, n]    
                    # compute correlation  
                    w = np.copy(W[n, :].T ) 
                    y_tilde = np.copy(w.T.dot(X))
                    w = w.reshape((-1, 1))
 
                    e_pair = epsilon(y_tilde.T, r_n_c )  
                    # abs of similarity measure
                    dis_wr = np.abs(e_pair) 

                   # choose rho_n based on update scheme 
                    if mu_idx[n] == 1: 
                        if np.size(rho[rho < dis_wr]) != 0:
                            rho_n = np.max(rho[rho < dis_wr])
                    else: 
                        if np.size(rho[rho > dis_wr]) != 0:
                            rho_n = np.min(rho[rho > dis_wr])

                    if rho.size == 0:
                        rho_n = 0.01 
                                
                    # store rho 
                    rho_n_arr[iter, n] = np.copy(e_pair)
                    # update mu 
                    mu_old[n] = np.copy(mu_c[n]) 

                    mu_idx[n] = mu_idx[n] or (mu_c[n] >= 1)
                    mu_idx[n] = mu_idx[n] and (mu_c[n] > 0) 
                    mu_c[n] = np.minimum(1, mu_c[n])
                    mu_c[n] = np.maximum(0, mu_c[n] + gam * (rho_n - dis_wr))    
                        
        

                if stochastic_search == 1:
                    weight = np.random.rand(1, T)
                else:
                    weight = np.ones((1, T))

                if max_i == 0:
                    EGx[0] = np.maximum(np.minimum(EGx[0], nf1['max_EGx']), nf1['min_EGx'])
                    grad = h / (np.dot(h.T, v)) + Z[:, :, n].dot((4* weight*y*yy).T) * simplified_ppval(nf1['pp_slope'], EGx[0]) / np.sum(weight)
                if max_i == 2:
                    EGx[2] = np.maximum(np.minimum(EGx[2], nf3['max_EGx']), nf3['min_EGx'])
                    grad = h / (np.dot(h.T, v)) + Z[:, :, n].dot((weight* sign_y*inv_pabs_y**2).T) * simplified_ppval(nf3['pp_slope'], EGx[2]) / np.sum(weight)
                if max_i == 4:
                    EGx[4] = np.maximum(np.minimum(EGx[4], nf5['max_EGx']), nf5['min_EGx'])
                    grad = h / (np.dot(h.T, v)) + Z[:, :, n].dot((weight* abs_y*(20+abs_y)*inv_p10abs_y**2).T) * simplified_ppval(nf5['pp_slope'], EGx[4]) / np.sum(weight)
                if max_i == 6:
                    EGx[6] = np.maximum(np.minimum(EGx[6], nf7['max_EGx']), nf7['min_EGx'])
                    grad = h / (np.dot(h.T, v)) + Z[:, :, n].dot((weight*(1-yy)*inv_pabs_yy**2).T) * simplified_ppval(nf7['pp_slope'], EGx[6]) / np.sum(weight)

                # Constant direction
                cnstd = Rz[:, :, n].dot(v)

                if n < num_guess:  
                    constraint_grad = epsilon_grad(r_n_c) 
                    grad = grad + constraint_grad


                # projected gradient
                grad =  grad - (cnstd.T.dot(grad) * cnstd /(np.dot(cnstd.T, cnstd))).reshape(-1, 1)
                check = inv_sqrtmH(Rz[:, :, n])
                grad = check.dot(grad)

                # Normalized gradient
                grad = grad / np.sqrt(grad.T.dot(Rz[:, :, n].dot(grad)))

                v = v.reshape(-1,1) + mu * grad


                if constraint == 'phase_retrieval' and stochastic_search == 1:
                    if n < amplitude.shape[1]:
                        v_tilde = pr_update(amplitude[:, n], y, a[:, n], Z[:, :, n])
                        v_tilde = np.reshape(v_tilde, v.shape)
                        v =  0.8* v +  0.2* v_tilde

                W[n, :] = np.copy(v.T )

  
            Cost[iter] = Cost[iter] - (np.sum(np.power(mu_c, 2 ) - np.power(mu_old, 2)) / (2 * gam ))
       
    
            if Cost[iter-1]  < min_cost:
                cost_increase_counter = 0
                min_cost = np.copy(Cost[iter-1])
                best_W = np.copy(last_W)
                best_a = np.copy(a)
            else:
                cost_increase_counter = cost_increase_counter + 1

            min_cost_queue[iter-1] = np.copy(min_cost)

            if cost_increase_counter > max_cost_increase:
                if stochastic_search == 1:
                    W1 = np.copy(W)
                    last_W1 = np.copy(last_W)
                    for n in range(N):
                        W1[n, :] = W1[n, :] / np.linalg.norm(W1[n, :])
                        last_W1[n, :] = last_W1[n, :] / np.linalg.norm(last_W1[n, :])
                    if 1 - np.min(np.abs(np.diag(np.dot(W1, last_W1.T)))) < tolerance:
                        break
                    else:
                        mu = np.copy(np.maximum((decaying_factor**(iter + 1)) , min_mu))
                        W = np.copy(best_W)
                        last_W = np.copy(best_W)
                        a = np.copy(best_a)
                        cost_increase_counter = 0
                        continue
                else:
                    W1 = np.copy(W)
                    last_W1 = np.copy(last_W)
                    for n in range(N):
                        W1[n, :] = W1[n, :] / np.linalg.norm(W1[n, :])
                        last_W1[n, :] = last_W1[n, :] / np.linalg.norm(last_W1[n, :])
                    if 1 - np.min(np.abs(np.diag(np.dot(W1, last_W1.T)))) < tolerance:
                        break
                    else:
                        mu = np.copy(np.maximum((decaying_factor**(iter + 1)) , min_mu))
                        W = np.copy(best_W)
                        last_W = np.copy(best_W)
                        a = np.copy(best_a)
                        cost_increase_counter = 0
                        continue

            last_W = np.copy(W)

        W = np.copy(best_W)
    W = np.dot(W, P)
    
    return W 


###############################################################################################################
# These functions are used in the ERBM algorithm.
###############################################################################################################


def lfc(x: np.ndarray, p: int , choice, a0) -> tuple[np.ndarray, np.ndarray]:
    """Helper function for ERBM ICA: computes the linear filtering coefficients (LFC) with length p for entropy rate estimation, and the estimated entropy rate.

    Args:
        x (np.ndarray, (Time Points, 1)): the source estimate [T x 1]
        p (int):  the filter length for the source model
        choice :  can be 'sub', 'super' or 'unknown'; any other input is handled as 'unknown' 
        a0 (np.ndarray or empty list): is the intial guess [p x 1] or an empty list []     

    Returns:
        a (np.ndarray, (p, 1)): the filter coefficients [p x 1]
        min_cost (np.ndarray, (1, 1)): the entropy rate estimation [1 x 1]
    """

    global nf1, nf3, nf5, nf7, cosmtx, sinmtx, Simpson_c

    tolerance = 1e-4
    T = x.shape[0]
    X0 = sp.linalg.convolution_matrix(x, p, 'full').T
    # remove tail so outliers have less effect
    X = X0[:, : T ]
    # remove DC
    X = X - np.mean(X, axis = 1).reshape(-1, 1)
    # pre-whitening
    R = np.dot(X, X.T) / T
    D, V = np.linalg.eig(R)
    order = np.argsort(D)
    d = D[order]
    V = V[:, order]
    eps = np.finfo(np.float64).eps
    d[d < 10 * eps]= 10 * eps
    P = np.dot(np.dot(V, np.diag(1/np.sqrt(d))), V.T)
    X = np.dot(P, X)

    if np.size(a0) == 0:
        # use SEA to provide the initial guess
        if choice == 'sub':
            # we don't need this case
            # TO DO
            pass
        if choice == 'super':
            # TO DO
            pass
        else:
            a = np.random.rand(p,1)
            a = a / np.linalg.norm(a)
            last_a = np.copy(a)
            for iter in range(100):
                y = np.dot(a.T, X)
                a = X.dot((y**3).T) / T - 3 * a
                a = np.copy(a / np.linalg.norm(a))
                if 1 - np.abs(a.T.dot(last_a)) < tolerance:
                    break
                else:
                    last_a = np.copy(a)

    else:
        a = np.linalg.solve(P, a0)

    min_cost = np.inf
    K = 8 # number of measuring functions
    best_a = np.copy(a)
    last_a = np.copy(a)
    min_mu = 1/128
    if np.size(a0) == 0:
        max_iter = 100
        mu = 4* min_mu
    else:
        max_iter = 100
        mu = 16* min_mu
    cost_increase_counter = 0
    Cost = np.zeros((max_iter, 1))
    
    for iter in range(max_iter):
        a = np.copy(np.reshape(a, (-1, 1)) )
        a_original = np.copy(P.dot(a))
        b_original, G_original = cnstd_and_gain(a_original)

        a = a.dot(np.exp(- G_original/2))
        b = P.dot(b_original)
        y = np.copy(np.dot(a.T, X))
        sigma2 = np.dot(a.T, a)
        # normalized y
        y = np.copy(y / np.sqrt(sigma2))

        Cost[iter] = np.copy(0.5 * np.log(2 * np.pi * sigma2) + 0.5)

        NE_Bound = np.zeros((K, 1))
        EGx = np.zeros((K, 1))
        # we only need to calculate these quantities once
        yy = y* y
        sign_y = np.sign(y)
        abs_y = np.abs(y)
        inv_pabs_y = 1/(1 + abs_y)
        inv_pabs_yy = 1/(1+ yy)
        inv_p10abs_y = 1/(10+abs_y)

        # G1(x) = x^4
        EGx[0] = np.sum(yy*yy)/T
        if EGx[0] < nf1['min_EGx']:
            NE_Bound[0] = simplified_ppval(nf1['pp_slope'], nf1['min_EGx'] ) * (EGx[0] - nf1['min_EGx'])
            NE_Bound[0] = simplified_ppval(nf1['pp'],nf1['min_EGx'])  + np.abs(NE_Bound[0] )
        else:
            if EGx[0] > nf1['max_EGx']:
                NE_Bound[0] = 0
            else:
                NE_Bound[0] = simplified_ppval(nf1['pp'], EGx[0] )

        # G3(x) = np.abs(x)/ (1 + np.abs(x))
        EGx[2] = 1 - np.sum(inv_pabs_y)/T
        if EGx[2] < nf3['min_EGx']:
            NE_Bound[2] = simplified_ppval(nf3['pp_slope'], nf3['min_EGx'] ) * (EGx[2] - nf3['min_EGx'])
            NE_Bound[2] = simplified_ppval(nf3['pp'], nf3['min_EGx']) + np.abs(NE_Bound[2])
        else:
            if EGx[2] > nf3['max_EGx']:
                NE_Bound[2] = simplified_ppval(nf3['pp_slope'], nf3['max_EGx'] ) * (EGx[2] - nf3['max_EGx'])
                NE_Bound[2] = simplified_ppval(nf3['pp'], nf3['max_EGx']) + np.abs(NE_Bound[2])

            else:
                NE_Bound[2] = simplified_ppval(nf3['pp'], EGx[2] )

        # G5(x)  = x* np.abs(x) /(10 + np.abs(x))
        EGx[4] = np.sum( y * abs_y * inv_p10abs_y )/T
        if EGx[4] < nf5['min_EGx']:
            NE_Bound[4] = simplified_ppval(nf5['pp_slope'], nf5['min_EGx'] ) * (EGx[4] - nf5['min_EGx'])
            NE_Bound[4] = simplified_ppval(nf5['pp'], nf5['min_EGx']) + np.abs(NE_Bound[4])
        else:
            if EGx[4] > nf5['max_EGx']:
                NE_Bound[4] = simplified_ppval(nf5['pp_slope'], nf5['max_EGx'] ) * (EGx[4] - nf5['max_EGx'])
                NE_Bound[4] = simplified_ppval(nf5['pp'], nf5['max_EGx']) + np.abs(NE_Bound[4])
            else:
                NE_Bound[4] = simplified_ppval(nf5['pp'], EGx[4] )

        # G7(x) =  x / (1 + x**2)
        EGx[6] = np.sum(y*inv_pabs_yy)/T
        if EGx[6] < nf7['min_EGx']:
            NE_Bound[6] = simplified_ppval(nf7['pp_slope'], nf7['min_EGx'] ) * (EGx[6] - nf7['min_EGx'])
            NE_Bound[6] = simplified_ppval(nf7['pp'], nf7['min_EGx']) + np.abs(NE_Bound[6])
        else:
            if EGx[6] > nf7['max_EGx']:
                NE_Bound[6] = simplified_ppval(nf7['pp_slope'], nf7['max_EGx'] ) * (EGx[6] - nf7['max_EGx'])
                NE_Bound[6] = simplified_ppval(nf7['pp'], nf7['max_EGx']) + np.abs(NE_Bound[6])
            else:
                NE_Bound[6] = simplified_ppval(nf7['pp'], EGx[6] )

        # select the tightest upper bound
        max_NE, max_i = np.max(NE_Bound), np.argmax(NE_Bound)
        Cost[iter] = np.copy(Cost[iter] - max_NE)
        last_a = np.copy(a)

        if Cost[iter] < min_cost:
            cost_increase_counter = 0
            min_cost = np.copy(Cost[iter])
            best_a = np.copy(a)
        else:
            cost_increase_counter = cost_increase_counter + 1

        if cost_increase_counter > 0: 
            if mu > min_mu:
                mu = mu / 2
                cost_increase_counter = 0
                a = np.copy(best_a)
                last_a = np.copy(best_a)
                continue
            else:
                break

        grad = a / sigma2
        if max_i == 0:
            EGx[0] = np.maximum(np.minimum(EGx[0], nf1['max_EGx']), nf1['min_EGx'])
            grad = grad - X.dot((4*y * yy).T) * simplified_ppval(nf1['pp_slope'], EGx[0]) / T /np.sqrt(sigma2)
            grad = grad + np.sum(4* y* yy* y ) * simplified_ppval(nf1['pp_slope'], EGx[0])* a / T / sigma2
        if max_i == 2:
            EGx[2] = np.maximum(np.minimum(EGx[2], nf3['max_EGx']), nf3['min_EGx'])
            grad = grad - X.dot( sign_y *inv_pabs_y**2) * simplified_ppval(nf3['pp_slope'], EGx[2]) / T / np.sqrt(sigma2)
            grad = grad + np.sum(sign_y*inv_pabs_y**2*y) * simplified_ppval(nf3['pp_slope'], EGx[2]) * a / T / sigma2
        if max_i == 4:
            EGx[4] = np.maximum(np.minimum(EGx[4], nf5['max_EGx']), nf5['min_EGx'])
            grad = grad - X.dot( abs_y*(20+abs_y)*inv_p10abs_y**2) * simplified_ppval(nf5['pp_slope'], EGx[4]) / T / np.sqrt(sigma2)
            grad = grad + np.sum( abs_y*(20+abs_y)*inv_p10abs_y**2*y ) * simplified_ppval(nf5['pp_slope'], EGx[4]) * a / T / sigma2
        if max_i == 6:
            EGx[6] = np.maximum(np.minimum(EGx[6], nf7['max_EGx']), nf7['min_EGx'])
            grad = grad - X.dot( (1-yy)*inv_pabs_yy**2) * simplified_ppval(nf7['pp_slope'], EGx[6]) / T / np.sqrt(sigma2)
            grad = grad + np.sum( (1-yy)*inv_pabs_yy**2*y) * simplified_ppval(nf7['pp_slope'], EGx[6]) * a / T / sigma2


        grad = grad- np.reshape(np.dot(grad.T, b)*b/(np.dot(b.T, b)) , (1, -1))
        grad = np.sqrt(sigma2) * grad/ np.linalg.norm(grad)
        a = np.copy(a - mu * grad)

    a = np.reshape(a ,(-1, 1))
    a = np.copy(best_a)
    a = np.dot(P,a)

    return a, min_cost


def simplified_ppval(pp: dict, xs: float) -> float:
    """Helper function for ERBM ICA: simplified version of ppval.
        This function evaluates a piecewise polynomial at a specific point. 
    
    Args:
        pp (dict): a dictionary containing the piecewise polynomial representation of a function
        xs (float): the evaluation point

    Returns:
        v (float): the value of the function at xs   
    """

    b = pp['breaks'][0]
    c = pp['coefs']
    l_pieces = int(pp['pieces'] )
    k = 4
    # find index
    index = float('nan ')
    middle_index = float('nan ')
    if xs > b[l_pieces-1]:
        index = l_pieces-1
    else:
        if xs < b[1]:
            index = 0
        else :
            low_index = 0
            high_index = l_pieces-1

            while True :
                middle_index = int(np.ceil(((0.6* low_index + 0.4* high_index))))
                if xs < b[middle_index]:
                    high_index = middle_index
                else:
                    low_index = middle_index
                if low_index == high_index -1:
                    index = low_index
                    break
    # now go to local coordinates
    xs = xs - b[index]
    # nested multiplication
    v = c[index, 0]
    for i in range(1, k ):
        v = v*xs + c[index, i]
    return v

def cnstd_and_gain(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Helper function for ERBM ICA: returns constraint direction used for calculating projected gradient and gain of filter a.
    
    Args:
        a (np.ndarray, (p, 1)): the filter coefficients [p x 1]
    
    Returns:
        b (np.ndarray, (p, 1)): the constraint direction [p x 1]
        G (np.ndarray, (1,)): the gain of the filter a
    """

    global cosmtx, sinmtx, Simpson_c


    eps = np.finfo(np.float64).eps
    p = a.shape[0]
    # calculate the integral
    # sample omega from 0 to pi
    n = 10*p
    h = np.pi / n

    # calculate |A(w)|^2
    Awr = np.zeros((1, n+1))  # real part
    Awi = np.zeros((1, n+1))  # imaginary part
    for q in range(p):
        Awr = Awr + a[q] * cosmtx[q, :]
        Awi = Awi + a[q] * sinmtx[q, :]

    Aw2 = 10*eps+ Awr**2 + Awi**2

    # calculate the vector
    v = np.zeros((p+1, n+1))
    inv_Aw2 = 1 / Aw2
    for q in range(p):
        v[q, :] = cosmtx[q, :] * inv_Aw2
    v[p,:] = np.log(Aw2)/np.pi

    # this is the integral
    u = h * v.dot(Simpson_c/3)
    b = sp.linalg.toeplitz(u[:p].ravel()).dot(a)

    # gain
    G = u[p]
    return b, G



def calculate_cos_sin_mtx(p: int) -> None :
    """Helper function for ERBM ICA: calculates the cos and sin matrix for integral calculation in ERBM ICA.
    
    Args:
        p (int): the filter length for the invertible filter source model   
    
    Returns:
        None
    """

    # prepare the cos and sin matrix for integral calculation
    global cosmtx, sinmtx, Simpson_c

    # sample omega from 0 to pi
    n = 10*p
    h = np.pi / n
    omega = np.arange(0, n+1, 1) * h

    cosmtx = np.zeros((p, n+1))
    sinmtx = np.zeros((p, n+1))
    for q in range(p):
        cosmtx[q, :] = np.cos(q * omega)
        sinmtx[q, :] = np.sin(q * omega)
    # c ist the vetcor used in Simpson's rule
    Simpson_c = np.zeros((n+1, 1))
    Simpson_c[np.arange(0, n+1, 2)] = 2
    Simpson_c[np.arange(1, n, 2)] = 4
    Simpson_c[0] = 1
    Simpson_c[n] = 1


def pre_processing(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Helper function for ERBM ICA: Preprocessing (removal of mean, patial pre-whitening, temporal pre-filtering)
    
    Args:
        X (np.ndarray, (Channels, Time Points)): the [N x T] input multivariate time series with dimensionality N observations/channels and T time points
    
    Returns:
        X (np.ndarray, (Channels, Time Points)): the pre-processed input multivariate time series
        P (np.ndarray, (Channels, Channels)): the pre-whitening matrix
    """
    # pre-processing of the data
    N, T = X.shape
    # remove mean
    X = X - np.mean(X, axis = 1).reshape(N, 1)
    # spatio pre-whitening
    R = np.dot(X, X.T) / T
    P1 = inv_sqrtmH(R)
    X = np.dot(P1, X)
    # temporal pre-filtering for colored signals
    q = 3
    r = np.zeros((q, 1))
    for  p in range(q):
        r[p] = np.trace(X[:, : T-p].dot(X[:, p: T].T)) / T / N

    af  = np.linalg.solve(sp.linalg.toeplitz(r[:q-1].ravel()), np.conjugate(r[1:]) )
    for n in range(N):
        X[n, :] =  sp.signal.lfilter(np.concatenate((np.ones((1,1)), -af), axis = 0)[:,0], 1 ,X[n, :])

    # spatio pre-whitening
    R = np.dot(X, X.T) / T
    P2 = inv_sqrtmH(R)
    X = np.dot(P2, X)
    P = np.dot(P2, P1)

    return X, P

def inv_sqrtmH(B: np.ndarray) -> np.ndarray:
    """Helper function for ERBM ICA: computes the inverse square root of a matrix.
    
    Args:
        B (np.ndarray): a square matrix
        
    Returns:
        A (np.ndarray): the inverse square root of B 
    """
    D, V = np.linalg.eig(B)
    order = np.argsort(D)
    D = D[order]
    V = V[:, order]
    #print('D', D)
    d = 1/np.sqrt(D)
    A = np.dot(np.dot(V, np.diag(d)), V.T)
    return A

def auction(assignCost, guard=None):
    """
    Performs assignment using Bertsekas' auction algorithm.

    Parameters:
    assignCost (ndarray): m x n matrix of costs for associating each row with each column. m >= n.
    guard (float, optional): Cost of column non-assignment. All assignments will have cost < guard.
    
    Returns:
    colsol (ndarray): Column assignments, where colsol[j] gives the row assigned to column j.
    rowsol (ndarray): Row assignments, where rowsol[i] gives the column assigned to row i.
    """
    
    m, n = assignCost.shape

    if m < n:
        raise ValueError('Cost matrix must have no more columns than rows.')

    # Augment cost matrix with a guard row if specified.
    m0 = m
    if guard is not None and np.isfinite(guard):
        m += 1
        assignCost = np.vstack((assignCost, np.full((1, n), guard)))

    # Initialize return arrays
    colsol = np.zeros(n, dtype=int)
    rowsol = np.zeros(m, dtype=int)
    price = np.zeros(m)
    EPS = np.sqrt(np.finfo(float).eps) / (n + 1)

    # 1st step is a full parallel solution. Get bids for all columns
    jp = np.arange(n)
    f = assignCost.copy()
    b1 = np.min(f, axis=0)  # cost of the best choice for each column
    ip = np.argmin(f, axis=0)  # row index of the best choice for each column
    f[ip, jp] = np.inf  # eliminate the best from contention

    bids = np.min(f, axis=0) - b1  # cost of runner-up choice hence bid
    ibid = np.argsort(bids)  # Arrange bids so highest are last

    # Now assign best bids (lesser bids are overwritten by better ones).
    price[ip[ibid]] += EPS + bids[ibid]
    rowsol[ip[ibid]] = jp[ibid] + 1  # +1 to convert to 1-based indexing
    iy = np.nonzero(rowsol)[0]
    colsol[rowsol[iy] - 1] = iy + 1  # -1 to convert back to 0-based indexing for Python

    # The guard row cannot be assigned (always available)
    if m0 < m:
        price[m - 1] = 0
        rowsol[m - 1] = 0

    # Now Continue with non-parallel code handling any contentions.
    while not np.all(colsol):
        for jp in np.where(colsol == 0)[0]:
            f = assignCost[:, jp] + price  # costs
            b1 = np.min(f)  # cost and row of the best choice
            ip = np.argmin(f)
            if ip >= m0:
                colsol[jp] = m
            else:
                f[ip] = np.inf  # eliminate from contention
                price[ip] += EPS + np.min(f) - b1  # runner-up choice hence bid
                if rowsol[ip]:  # take the row away if already assigned
                    colsol[rowsol[ip] - 1] = 0
                rowsol[ip] = jp + 1  # +1 to convert to 1-based indexing
                colsol[jp] = ip + 1  # +1 to convert to 1-based indexing

    # Screen out infeasible assignments
    if m > m0:
        colsol[colsol == m] = 0
        rowsol = rowsol[:m0]

    return colsol - 1, rowsol - 1  # -1 to convert back to 0-based indexing for Python



   