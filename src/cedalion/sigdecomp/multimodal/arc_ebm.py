"""Independent Component Analysis by Entropy Bound Minimization (ICA-EBM).

This code is based on :cite:t:`Li2010A` and converted matlab versions provided by the
MLSP Lab at the University of Maryland, which is available here:
https://mlsp.umbc.edu/resources.html.


(:cite:t:`yang2025flexible`) H. Yang, T. Vu, Ehsan Ahmed Dhrubo, V. D. Calhoun, and Tülay Adali, 
“A Flexible Constrained ICA Approach for Multisubject fMRI Analysis,” 
International Journal of Biomedical Imaging, vol. 2025, no. 1, Jan. 2025, 
doi: https://doi.org/10.1155/ijbi/2064944.
"""


import numpy as np
import cedalion.data
from cedalion import cite

def arc_ebm(X: np.ndarray, guess_mat, constraint = 'correlation') -> np.ndarray:
    """Adaptive-reverse Constrained ICA by Entropy Bound Minimization (arc-EBM) is a constrained ICA algorithm.
        arc-EBM calculates the blind source separation demixing matrix corresponding to X, 
        using the reference signals in guess_mat and the constraint specified by constraint.

    Args:
        X (np.ndarray, (Channels, Time Points)): the [N x T] input multivariate time series with dimensionality N observations/channels and T time points
        guess_mat (np.ndarray, (Time Points, Referenced Channels)), (np.ndarray, (Time Points/2, Referenced Channels)): Time or frequency domain reference signals. The number of reference signals should be less than or equal to the number of channels in X. The first dimension should be T for time domain signals and T/2 for frequency domain signals.
        constraint (str): the constraint to be used for the gradient step, either 'correlation' (default) or 'psd'

    Returns:
        W (np.ndarray, (Channels, Channels)): the [N x N] demixing matrix with weights for  N channels/sources. 
            To obtain the independent components, the demixed signals can be calculated as S = W @ X.
    
    Initial Contributors:
        - Jacqueline Behrendt | j.behrendt@tu-berlin.de | 2026

    References:
    This code is based on the matlab version by Xi-Lin Li (:cite:t:`Li2010A`)
        Xi-Lin Li and Tulay Adali, "Independent component analysis by entropy bound minimization," 
        IEEE Trans. Signal Processing, vol. 58, no. 10, pp. 5151-5164, Oct. 2010.

        (:cite:t:`yang2025flexible`) H. Yang, T. Vu, Ehsan Ahmed Dhrubo, V. D. Calhoun, and Tülay Adali, 
        “A Flexible Constrained ICA Approach for Multisubject fMRI Analysis,” 
        International Journal of Biomedical Imaging, vol. 2025, no. 1, Jan. 2025, 
        doi: https://doi.org/10.1155/ijbi/2064944.

    """

    cite("Li2010A")
    cite("yang2025flexible")
    ###############################################################################################################
    # Part 0: Preprocessing
    ###############################################################################################################
    
    rho = np.arange(0, 1.01, 0.01)  
    max_iter_fastica = 100
    max_iter_orth = 1000
    max_iter_orth_refine = 1000
    max_iter_nonorth = 1000
    saddle_test_enable = True
    tolerance = 1e-6
    max_cost_increase_number = 5
    stochastic_search_factor = 1
    eps = np.finfo(np.float64).eps 
    gam = 100 

    # report the progress if verbose == True
    verbose = False

    # Load 8 measuring functions. But we only use 4 of them.
    K = 8
    file_path = cedalion.data.get("measfunc_table.npy")
    table = np.load(file_path, allow_pickle=True)

    nf1, nf3, nf5, nf7 = table[0], table[2], table[4], table[6] 

    N = X.shape[0] # number of channels
    T = X.shape[1] # number of time points  
    X, P = pre_processing(X)

    # Define the epsilon function based on the constraint
    if constraint == 'correlation':   
        def epsilon(a,b): 
            # correlation coefficient   
            return  np.corrcoef(a, b)[0, 1] 
        
        def epsilon_grad():  
            # gradient for correlation constraint   
            mu_signed[n] = np.sign(e_pair) *  mu_c[n]
            c_grad = mu_signed[n] * (X.dot(r_n_c) ) * (1/np.sqrt(T))
            return np.reshape(c_grad, (-1, 1))
    

    if constraint == 'psd': 
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

        # define correlation between psd of estimated source and reference psd
        def epsilon(a,b): 
            # power spectral density 
            # compute psd for real signal a
            # b is already a psd 
            psd_a = (2/T) * np.abs(np.fft.rfft(a))**2  
            psd_correlation = np.corrcoef(psd_a, b)[0,1]
            return psd_correlation 

        # define dot product between psd of estimated source and reference psd
        def epsilon_dot(a,b): 
            # abs of dot product 
            a = a / np.linalg.norm(a)   
            psd_a = (2/T) * np.abs(np.fft.rfft(a))**2 
            psd_a = psd_a - np.mean(psd_a)   
            b = b - np.mean(b)   
            b = b / np.linalg.norm(b)     
            abs_dot = np.abs(np.dot(psd_a, b))     
            return abs_dot

        # define gradient function for psd constraint
        def epsilon_grad(): 
            # compute gradient of epsilon for the psd constraint   
            psd_s = (2/T) * np.abs(np.fft.rfft(w.T.dot(X)))**2     
            sign = np.sign(np.corrcoef(psd_s, r_n_c)[0,1])
            r = np.reshape(r_n_c, (-1, 1, 1))    
            c_grad  = sign * mu_c[n] * np.sum( np.multiply(np.dot(C_tilde, w), r), axis = 0  )    
            return c_grad
        

    # make initial guess for demixing matrix W
    W = np.random.rand(N, N)

    # symmetric decorrelation    
    W = symdecor(W)   

    num_guess = guess_mat.shape[1] # number of reference signals    
    mu_c = np.ones((num_guess, 1))  
    corr_w_guess = np.zeros((num_guess, N)) 
    num_W = np.shape(W)[0]  
    corr_w_guess = np.zeros((num_guess, num_W)) 

    # resort W based on correlation with reference signals 
    for kl in range(num_guess): 
        r_n_c = guess_mat[:, kl]    
        for lp in range(num_W): 
            w = W[lp, :].T
            corr_w_guess[kl, lp] = epsilon(X.T.dot(w), r_n_c)  
     

    # may need auction to auction to choose order 
    max_index = np.argmax(np.abs(corr_w_guess), axis = 1)  
    if len(np.unique(max_index)) != num_guess: 
        colsol, _ = auction((1- np.abs(corr_w_guess)).T)
        max_index = colsol.T

    c = np.arange(0, num_W) 
    c = np.setdiff1d(c, max_index)  
    sort_order = np.concatenate((max_index, c)) 
    W = W[sort_order, :] 

    last_W = np.copy(W) 
    best_W = np.copy(W) 
    Cost = np.zeros((max_iter_fastica, 1))  
    min_cost = np.inf
    cost_increaser_counter = 0 
    negentropy_array = np.zeros((N,1 ))  


    for iter in range(max_iter_fastica): 
        Y = np.copy(W.dot(X))
        for n in range(N): 
            y = np.copy(Y[n, :])
            # evaluate the upper bound of negentropy of the nth component 
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
            Cost[iter] = np.copy(Cost[iter] - max_NE)


        if Cost[iter] < min_cost:
            min_cost = np.copy(Cost[iter])
            best_W = np.copy(last_W)
            cost_increaser_counter = 0
        else:
            cost_increaser_counter = cost_increaser_counter + 1

        W = np.multiply(np.multiply(Y, Y), Y).dot(X.T) / T - 3 * W
        W = symdecor(W)


        if 1 - np.min(np.abs(np.diag(W.dot(last_W.T)))) < tolerance: 
            break 
        else : 
            last_W = np.copy(W) 
        if cost_increaser_counter > max_cost_increase_number: 
            break 

    W = np.copy(best_W)    
##############################################################################################################
#     Part 1: Orthogonal ICA    
#   varying step size, stochastic gradient search
##############################################################################################################

    if verbose:
        print('Orthogonal ICA stage.')


    # resort existing W based on correlation with reference signals 
    for kl in range(num_guess): 
        r_n_c = guess_mat[:, kl]    
        for lp in range(num_W): 
            w = W[lp, :].T 
            corr_w_guess[kl, lp] = epsilon(X.T.dot(w), r_n_c)    

    # may need auction to auction to choose order 
    max_index = np.argmax(np.abs(corr_w_guess), axis = 1)  

    if len(np.unique(max_index)) != num_guess:   
        colsol, _ = auction((1- np.abs(corr_w_guess)).T)
        max_index = colsol.T

    c = np.arange(0, num_W) 
    c = np.setdiff1d(c, max_index)  
    sort_order = np.concatenate((max_index, c))   

    W = W[sort_order, :]  


    last_W = np.copy(W)
    best_W = np.copy(W)
    Cost = np.zeros((max_iter_orth, 1)) 
    min_cost = np.inf
    min_cost_queue = np.copy(min_cost* np.ones((max_iter_orth, 1)))
    mu = 1/6.25
    min_mu = 1/50 
    cost_increaser_counter = 0 
    fastica_on = True   
    max_negentropy = np.zeros((N, 1))
    negentropy_array = np.zeros((N, 1))    

    for iter in range(max_iter_orth):   
        Y = np.copy(W.dot(X))  
        for n in range(N):  
            w = np.copy(W[n, :].T)
            y = np.copy(Y[n, :] )

            # evaluate the upper bound of negentropy of the nth component   
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
            Cost[iter] = np.copy(Cost[iter] - max_NE)

            if ~fastica_on: 
                weight = np.random.rand(1, T)    

            # Perform orthogonal ICA   
            if max_i == 0:
                # G1(x) = x^4
                if fastica_on :  
                    grad = X.dot( (4* y* yy).T )/T 
                    Edgx = 12 
                else : 
                    grad = X.dot((4 * weight * y * yy ).T ) / np.sum(weight)
                    vEGx = 2 * (EGx[0] > nf1['critical_point']) -1 
            elif max_i == 2:    
                # G3(x) = np.abs(x)/ (1 + np.abs(x))    
                if fastica_on : 
                    grad = X.dot( (sign_y * inv_pabs_y * inv_pabs_y).T )/T 
                    Edgx = np.sum(-2 * inv_pabs_y * inv_pabs_y * inv_pabs_y)/T   
                else :
                    grad = X.dot((weight * sign_y * inv_pabs_y * inv_pabs_y).T ) / np.sum(weight) 
                    vEGx = 2 * (EGx[2] > nf3['critical_point']) -1   
            elif max_i == 4:    
                # G5(x)  = x* np.abs(x) /(10 + np.abs(x))
                if fastica_on : 
                    grad  = X.dot((abs_y *(20 + abs_y) * inv_p10abs_y * inv_p10abs_y).T )/T
                    Edgx = np.sum(200 * sign_y * inv_p10abs_y * inv_p10abs_y * inv_p10abs_y)/T  
                else :
                    grad = X.dot((weight * abs_y * (20 + abs_y) * inv_p10abs_y**2 ).T ) / np.sum(weight) 
                    vEGx = 2 * (EGx[4] > nf5['critical_point']) -1  
            elif max_i == 6:    
                # G7(x) =  x / (1 + x**2)   
                if fastica_on : 
                    grad = X.dot(((1 - yy)* inv_pabs_yy**2).T )/T
                    Edgx = np.sum(2 * y * (yy-3)* inv_pabs_yy* inv_pabs_yy* inv_pabs_yy)/T
                else :  
                    grad = X.dot((weight * (1 - yy) * inv_pabs_yy**2 ).T ) / np.sum(weight) 
                    vEGx = 2 * (EGx[6] > nf7['critical_point']) -1 
            if fastica_on :
                w1 = grad - Edgx * w  
            else :
                grad = vEGx * grad  
                w = np.reshape(w, (-1, 1)) 
                grad = grad - ((w.T).dot(grad)) * w   
                grad = grad / np.linalg.norm(grad)  
                w1 = w + mu * grad 

            W[n, :] = np.copy(w1.T)

        W = np.copy(symdecor(W)) 

        if Cost[iter] < min_cost:
            cost_increaser_counter = 0  
            min_cost = np.copy(Cost[iter])
            best_W = np.copy(last_W)
            max_negentropy = np.copy(negentropy_array)
        else: 
            cost_increaser_counter = cost_increaser_counter + 1

        min_cost_queue[iter] = np.copy(min_cost)

        if fastica_on : 
            if cost_increaser_counter >= max_cost_increase_number  or 1- np.min(np.abs(np.diag(W.dot(last_W.T)))) < tolerance:   
                cost_increaser_counter = 0 
                W = np.copy(best_W ) 
                last_W = np.copy(W)
                fastica_on = False 
                continue
        else :  
            if cost_increaser_counter > stochastic_search_factor * max_cost_increase_number: 
                if mu > min_mu:
                    cost_increaser_counter = 0 
                    W = np.copy(best_W ) 
                    last_W = np.copy(W)
                    mu = mu/2  
                    continue 
                else: 
                    break
        last_W = np.copy(W)

    # End of Part 1  
    W = np.copy(best_W)
    ##############################################################################################################
    # Part 2: check for saddle points
    ##############################################################################################################
    if saddle_test_enable :
        if verbose: 
            print('Saddle point detection.')
        SADDLE_TESTED = False
        saddle_tested = True 

        while saddle_tested: 
            saddle_tested = False 
            Y = np.copy(W.dot(X))
            for m in range(N): 
                w1 = np.copy(W[m, :].T )
                ym = np.copy(Y[m, :])   
                for n in range(m+1, N): 
                    w2 = np.copy(W[n, :].T )
                    yn = np.copy(Y[n, :])

                    yr1 = (ym + yn)/ np.sqrt(2)
                    yr2 = (ym - yn)/ np.sqrt(2) 
                    y = np.copy(yr1)
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
                    negentropy1 = max_NE

                    y = np.copy(yr2)  
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
                    negentropy2 = max_NE

                    if negentropy1 + negentropy2 > max_negentropy[m] + max_negentropy[n]+ eps : 
                        if verbose:
                            print('Rotationg %d and %d.' % (m, n))
                        max_negentropy[m] = np.copy(negentropy1)
                        max_negentropy[n] = np.copy(negentropy2)
                        W[m, : ] = np.copy((w1+ w2).T/np.sqrt(2))
                        W[n, : ] = np.copy((w1- w2).T/np.sqrt(2) )
                        Y[m, :] = yr1   
                        Y[n, :] = yr2   
                        ym = yr1
                        w1 = np.copy(W[m, :].T  )
                        saddle_tested = True
                        SADDLE_TESTED = True


    else: 
        SADDLE_TESTED = False

    if SADDLE_TESTED : 
    ##############################################################################################################
    # Part 3: if saddle points are detected, refine orthogonal ICA
    # fix step size gradient search 
    ##############################################################################################################
        if verbose:
            print('Orthogonal ICA refinement is required because saddle points are detected.')
        last_W = np.copy(W) 
        best_W = np.copy(W) 
        Cost = np.zeros((max_iter_orth_refine, 1))  
        min_cost = np.inf   
        min_cost_queue = min_cost * np.ones((max_iter_orth_refine, 1))  
        mu = 1/ 50 
        cost_increaser_counter = 0  
        fastica_on = True 

        for iter in range(max_iter_orth_refine): 
            for n in range(N): 
                w = np.copy(W[n, :].T) 
                y = np.copy(w.T.dot(X)) 
                # evaluate the upper bound of negentropy of the nth component   
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
                negentropy_array[n] = max_NE
                Cost[iter] = np.copy(Cost[iter] - max_NE) 

                # Perform orthogonal ICA   
                if max_i == 0:
                    # G1(x) = x^4
                    if fastica_on :  
                        grad = X.dot( (4* y* yy).T )/T 
                        Edgx = 12 
                    else : 
                        grad = X.dot((4 * weight * y * yy ).T ) / np.sum(weight)
                        vEGx = 2 * (EGx[0] > nf1['critical_point']) -1 
                elif max_i == 2:    
                    # G3(x) = np.abs(x)/ (1 + np.abs(x))    
                    if fastica_on : 
                        grad = X.dot( (sign_y * inv_pabs_y * inv_pabs_y).T )/T   
                        Edgx = np.sum(-2 * inv_pabs_y * inv_pabs_y * inv_pabs_y)/T 
                    else :
                        grad = X.dot((weight * sign_y * inv_pabs_y * inv_pabs_y).T ) / np.sum(weight) 
                        vEGx = 2 * (EGx[2] > nf3['critical_point']) -1 
                elif max_i == 4:    
                    # G5(x)  = x* np.abs(x) /(10 + np.abs(x))
                    if fastica_on : 
                        grad  = X.dot((abs_y *(20 + abs_y) * inv_p10abs_y * inv_p10abs_y).T )/T
                        Edgx = np.sum(200 * sign_y * inv_p10abs_y * inv_p10abs_y * inv_p10abs_y)/T  
                    else :
                        grad = X.dot((weight * abs_y * (20 + abs_y) * inv_p10abs_y**2 ).T ) / np.sum(weight) 
                        vEGx = 2 * (EGx[4] > nf5['critical_point']) -1  
                elif max_i == 6:    
                    # G7(x) =  x / (1 + x**2)   
                    if fastica_on : 
                        grad = X.dot(((1 - yy)* inv_pabs_yy**2).T )/T
                        Edgx = np.sum(2 * y * (yy-3)* inv_pabs_yy* inv_pabs_yy* inv_pabs_yy)/T
                    else :  
                        grad = X.dot((weight * (1 - yy) * inv_pabs_yy**2 ).T ) / np.sum(weight) 
                        vEGx = 2 * (EGx[6] > nf7['critical_point']) -1 
                if fastica_on :
                    w1 = grad - Edgx * w  
                else :
                    grad = vEGx * grad  
                    w = np.reshape(w, (-1, 1)) 
                    grad = grad - ((w.T).dot(grad)) * w 
                    grad = grad / np.linalg.norm(grad)  
                    w1 = w + mu * grad 

                W[n, :] = np.copy(w1.T) 

            W = np.copy(symdecor(W) )

            if Cost[iter] < min_cost:
                cost_increaser_counter = 0  
                min_cost = np.copy(Cost[iter])
                best_W = np.copy(last_W)
                max_negentropy = np.copy(negentropy_array)
            else: 
                cost_increaser_counter = cost_increaser_counter + 1

            min_cost_queue[iter] = np.copy(min_cost)


            if fastica_on : 
                if cost_increaser_counter >= max_cost_increase_number  or 1- np.min(np.abs(np.diag(W.dot(last_W.T)))) < tolerance:   
                    cost_increaser_counter = 0 
                    W = np.copy(best_W ) 
                    last_W = np.copy(W)
                    fastica_on = False 
                    continue
            else :  
                if cost_increaser_counter > stochastic_search_factor * max_cost_increase_number:
                    break
            last_W = np.copy(W)

        W = np.copy(best_W) 
##############################################################################################################
# Part 4: non-orthogonal ICA 
# fix small step size for refinement, gradient search 
##############################################################################################################
    if verbose:
        print('Non-orthogonal ICA stage.')  

    # resort W based on correlation with reference signals 
    for kl in range(num_guess): 
        r_n_c = guess_mat[:, kl]    
        for lp in range(num_W): 
            w = W[lp, :].T
            corr_w_guess[kl, lp] = epsilon( X.T.dot(w), r_n_c)  


    # may need auction to to choose order 
    max_index = np.argmax(np.abs(corr_w_guess), axis = 1)    

    if len(np.unique(max_index)) != num_guess:   
        colsol, _ = auction((1- np.abs(corr_w_guess)).T)
        max_index = colsol.T

    c = np.arange(0, num_W) 
    c = np.setdiff1d(c, max_index)  
    sort_order = np.concatenate((max_index, c))   
    W = np.copy(W[sort_order, :]) 

    last_W = np.copy(W) 
    best_W = np.copy(W)
    Cost = np.zeros((max_iter_nonorth, 1)) 
    min_cost_queue = np.copy(min_cost * np.ones((max_iter_nonorth, 1)))
    mu = 1 
    min_mu = 1/200  
    max_cost_increase_number = 3 
    cost_increaser_counter = 0  
    mu_idx = np.full(mu_c.shape, False)

    decaying_factor = 0.95
    min_change = 1e-4 
    min_iter = 100
    mu_old = np.copy(mu_c)
    rho_n_arr = np.zeros((max_iter_nonorth, N))
    mu_signed = np.zeros((N, 1 ))    
    for iter in range(max_iter_nonorth): 
        Cost[iter] = np.copy(- np.log(np.abs(np.linalg.det(W))))

        for n in range(N):  
            if N > 7:  
                if n == 0: 
                    Wn = np.copy(W[1:N, :])  
                    inv_Q = np.copy(np.linalg.inv(Wn.dot(Wn.T)))
                else: 
                    n_last = np.copy(n-1)    
                    Wn_last = np.copy(np.delete(W, n_last, axis = 0)) 
                    w_current = np.copy(W[n, :].T ) 
                    w_last = np.copy(W[n_last, :].T) 

                    c = Wn_last.dot(w_last- w_current)  
                    c[n_last ] = 0.5* ((w_last.T).dot(w_last) - (w_current.T).dot(w_current) )
                    e_last = np.zeros((N-1, 1)) 
                    e_last[n_last] = 1  

                    temp1 = np.reshape(inv_Q.dot(c), (-1, 1 ))
                    temp2 = np.reshape(inv_Q[:, n_last ], (-1, 1))
                    inv_Q_plus = inv_Q - (temp1.dot(temp2.T) / (1 + temp1[n_last]))  

                    temp1 = np.reshape(inv_Q_plus.T.dot(c), (-1, 1))
                    temp2 = np.reshape(inv_Q_plus[:, n_last   ], (-1, 1 ))
                    inv_Q = inv_Q_plus - (temp1.dot(temp2.T) / (1 + c.T.dot(temp2)))
                    # make inv_Q hermitian
                    inv_Q = np.copy((inv_Q + inv_Q.T )/2 ) 



                temp1 = np.random.rand(N, 1) 
                W_n = np.copy(np.delete(W, n, axis = 0)) 
                h = temp1 - W_n.T.dot(inv_Q.dot(W_n.dot(temp1))) 

            else:
                temp1 = np.random.rand(N, 1) 
                temp2 = np.copy(np.delete(W, n, axis = 0) ) 
                h = temp1 - temp2.T.dot(np.linalg.inv(temp2.dot(temp2.T)).dot(temp2.dot(temp1)))   

            w = np.copy(W[n, :].T ) 
            y = np.copy(w.T.dot(X))

            # evaluate the upper bound of negentropy of the nth component   
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
            Cost[iter] = np.copy(Cost[iter] - max_NE ) 

            # Include constraint here: 

            if n < num_guess: 
                # choose reference signal
                r_n_c = guess_mat[:, n]    
                # compute correlation  
                if constraint == 'psd' :
                    e_pair = epsilon_dot(y.T, r_n_c )
                else :
                    e_pair = epsilon(y.T, r_n_c )   
                dis_wr = np.abs(e_pair) 

                if mu_idx[n] == 1: 
                    rho_n = np.max(rho[rho <= dis_wr])   
                else: 
                    rho_n = np.min(rho[rho > dis_wr])   
                
                if rho.size == 0:
                    rho_n = 0.01 
                # store rho 
                rho_n_arr[iter, n] = np.copy(rho_n)
                # update mu 
                mu_old[n] = np.copy(mu_c[n]) 

                mu_idx[n] = mu_idx[n] or (mu_c[n] >= 1)
                mu_idx[n] = mu_idx[n] and (mu_c[n] > 0) 
                mu_c[n] = np.minimum(1, mu_c[n])
                mu_c[n] = np.maximum(0, mu_c[n] + gam * (rho_n - dis_wr))   
    
                if constraint == 'psd' : 
                    r_n_c = r_n_c - np.mean(r_n_c) 

                r_n_c = r_n_c / np.linalg.norm(r_n_c)     


            if max_i == 0:
                # G1(x) = x^4
                vEGx = 2 * (EGx[0] > nf1['critical_point']) - 1
                grad = X.dot((4 * y * yy).T) / T
                EGx[0] = np.maximum(np.minimum(EGx[0], nf1['max_EGx']), nf1['min_EGx'])
                grad = (h / (h.T.dot(w))) + np.reshape(X.dot((4 * y * yy).T) * simplified_ppval(nf1['pp_slope'], EGx[0]) / T, (-1, 1))
            elif max_i == 2:
                # G3(x) = np.abs(x)/ (1 + np.abs(x))
                vEGx = 2 * (EGx[2] > nf3['critical_point']) - 1
                grad = X.dot((sign_y * inv_pabs_y * inv_pabs_y).T) / T
                EGx[2] = np.maximum(np.minimum(EGx[2], nf3['max_EGx']), nf3['min_EGx'])
                grad = (h / (h.T.dot(w))) + np.reshape(X.dot((sign_y * inv_pabs_y * inv_pabs_y).T) * simplified_ppval(nf3['pp_slope'], EGx[2]) / T, (-1, 1))
            elif max_i == 4:
                # G5(x)  = x* np.abs(x) /(10 + np.abs(x))
                vEGx = 2 * (EGx[4] > nf5['critical_point']) - 1
                grad = X.dot((abs_y * (20 + abs_y) * inv_p10abs_y * inv_p10abs_y).T) / T
                EGx[4] = np.maximum(np.minimum(EGx[4], nf5['max_EGx']), nf5['min_EGx'])
                grad = (h / (h.T.dot(w))) + np.reshape(X.dot((abs_y * (20 + abs_y) * inv_p10abs_y * inv_p10abs_y).T) * simplified_ppval(nf5['pp_slope'], EGx[4]) / T, (-1, 1))
            elif max_i == 6:
                # G7(x) =  x / (1 + x**2)
                vEGx = 2 * (EGx[6] > nf7['critical_point']) - 1
                grad = X.dot(((1 - yy) * inv_pabs_yy ** 2).T) / T
                EGx[6] = np.maximum(np.minimum(EGx[6], nf7['max_EGx']), nf7['min_EGx'])
                grad = (h / (h.T.dot(w))) + np.reshape(X.dot(((1 - yy) * inv_pabs_yy ** 2).T) * simplified_ppval(nf7['pp_slope'], EGx[6]) / T, (-1, 1))

            w = np.reshape(w, (-1, 1 ))

            # adapt gradient to include constraints
            if n < num_guess:   
                grad = grad + epsilon_grad()
   
            grad = grad - ((w.T).dot(grad)) * w 
            grad = grad / np.linalg.norm(grad)  
            w1 = w + mu * grad
            w1 = w1 / np.linalg.norm(w1)   
            W[n, :] = np.copy(w1.T  )


        Cost[iter] = Cost[iter]- (np.sum(np.power(mu_c, 2 ) - np.power(mu_old, 2)) / (2* gam ))
        mu = np.copy(np.maximum((decaying_factor**(iter + 1)) , min_mu))
        currentChange = np.maximum(0, np.max(1- np.abs(np.diag(last_W.dot(W.T)))))

        if currentChange < min_change and iter > min_iter:   
            best_W = np.copy(W) 
            break   
        else:
            last_W = np.copy(W) 
    
    W = best_W
    W = W.dot(P)   

    return W


###############################################################################################################
# These functions are used in the arc-EBM algorithm.
###############################################################################################################


def simplified_ppval(pp: dict, xs: float) -> float:
    """Helper function for ICA EBM: simplified version of ppval. 
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

def inv_sqrtmH(B: np.ndarray) -> np.ndarray:    
    """Helper function for ICA EBM: computes the inverse square root of a matrix.
    
    Args:
        B (np.ndarray): a square matrix
        
    Returns:    
        A (np.ndarray): the inverse square root of B 
    """

    D, V = np.linalg.eig(B) 
    order = np.argsort(D) 
    D = D[order]
    V = V[:, order]  
    d = 1/np.sqrt(D) 
    A = np.dot(np.dot(V, np.diag(d)), V.T)  
    return A

def pre_processing(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Helper function for ICA EBM: pre-processing (DC removal & spatial pre-whitening).
    
    Args:   
        X (np.ndarray, (Channels, Time Points) ): the data matrix [N x T] 
    
    Returns:    
        X (np.ndarray, (Channels, Time Points)): the pre-processed data matrix  [N x T] 
        P (np.ndarray, (Channels, Channels)): the pre-whitening matrix [N x N] 
    """ 

    # pre-processing program
    T = X.shape[1]
    # remove DC 
    Xmean = np.mean(X, axis=1) 
    X = X - np.tile(Xmean, (T, 1)).T
    # spatio pre-whitening
    R = np.dot(X, X.T) / T  
    P = inv_sqrtmH(R)
    X = np.dot(P, X)    
    return X, P

def symdecor(M: np.ndarray) -> np.ndarray: 
    """Helper function for ICA EBM: fast symmetric orthogonalization.
    
    Args:   
        M (np.ndarray, (Channels, Channels)): the matrix to be orthogonalized [N x N]

    Returns:    
        W (np.ndarray, (Channels, Channels)): the orthogonalized matrix [N x N]
    """

    D, V = np.linalg.eig(M.dot(M.T))    
    order = np.argsort(D)   
    D = D[order]
    V = V[:, order]    
    B = np.dot(np.ones((M.shape[1], 1)), np.reshape((1/np.sqrt(D)).T, (1, M.shape[1])   ))
    W = np.multiply(V, B ).dot(V.T.dot(M))
    return W


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

