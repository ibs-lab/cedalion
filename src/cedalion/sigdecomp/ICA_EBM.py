import scipy.io
import numpy as np
import matplotlib.pyplot as plt 

# References:
# This code is based on the matlab version by Xi-Lin Li. 
# Xi-Lin Li and Tulay Adali, "Independent component analysis by entropy bound minimization," 
# IEEE Trans. Signal Processing, vol. 58, no. 10, pp. 5151-5164, Oct. 2010. 

def ICA_EBM(X):
    # ICA-EBM: ICA by Entropy Bound Minimization (real-valued version)
    # Four nonlinearities
    # x^4,  |x|/(1+|x|),    x|x|/(10+|x|),  and     x/(1+x^2)
    # are used for entropy bound calculation 
    ###############################################################################################################
    # Inputs: 
    # X:    mixtures
    # Output:
    # W:    demixing matrix
    ###############################################################################################################
    # Part 0: Preprocessing 
    ###############################################################################################################
    max_iter_fastica = 100
    max_iter_orth = 1000
    max_iter_orth_refine = 1000
    max_iter_nonorth = 1000
    saddle_test_enable = True
    tolerance = 1e-4
    max_cost_increase_number = 5
    stochastic_search_factor = 1
    eps = np.finfo(np.float64).eps 

    verbose = False       # report the progress if verbose== True 
    show_cost = False    # show the cost values vs. iterations at each stage if show_cost== True  - not implemented yet 

    # Load 8 measuring functions. But we only use 4 of them.
    K = 8          
    table = np.load('measfunc_table.npy', allow_pickle= True)
    nf1, nf2, nf3, nf4, nf5, nf6, nf7, nf8 = table[0], table[1], table[2], table[3], table[4], table[5], table[6], table[7]
  

    N = X.shape[0]
    T = X.shape[1]
    X, P = pre_processing(X)
    
    # make initial guess 
    W = np.random.rand(N, N) 

    W = symdecor(W)    
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
    #if show_cost:    insert plots here
##############################################################################################################
#     Part 1: Orthogonal ICA    
#   varying step size, stochastic gradient search
##############################################################################################################       

    if verbose:
        print('Orthogonal ICA stage.')

    last_W = np.copy(W)
    best_W = np.copy(W)
    Cost = np.zeros((max_iter_orth, 1)) 
    min_cost = np.inf
    min_cost_queue = min_cost* np.ones((max_iter_orth, 1))
    mu = 1/6.25
    min_mu = 1/50 
    cost_increaser_counter = 0 
    fastica_on = True   
    error = 0 
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
                if fastica_on == True :  
                    grad = X.dot( (4* y* yy).T )/T 
                    Edgx = 12 
                else : 
                    grad = X.dot((4 * weight * y * yy ).T ) / np.sum(weight)
                    vEGx = 2 * (EGx[0] > nf1['critical_point']) -1 
            elif max_i == 2:    
                # G3(x) = np.abs(x)/ (1 + np.abs(x))    
                if fastica_on == True : 
                    grad = X.dot( (sign_y * inv_pabs_y * inv_pabs_y).T )/T 
                    Edgx = np.sum(-2 * inv_pabs_y * inv_pabs_y * inv_pabs_y)/T   
                else :
                    grad = X.dot((weight * sign_y * inv_pabs_y * inv_pabs_y).T ) / np.sum(weight) 
                    vEGx = 2 * (EGx[2] > nf3['critical_point']) -1   
            elif max_i == 4:    
                # G5(x)  = x* np.abs(x) /(10 + np.abs(x))
                if fastica_on == True : 
                    grad  = X.dot((abs_y *(20 + abs_y) * inv_p10abs_y * inv_p10abs_y).T )/T
                    Edgx = np.sum(200 * sign_y * inv_p10abs_y * inv_p10abs_y * inv_p10abs_y)/T  
                else :
                    grad = X.dot((weight * abs_y * (20 + abs_y) * inv_p10abs_y**2 ).T ) / np.sum(weight) 
                    vEGx = 2 * (EGx[4] > nf5['critical_point']) -1  
            elif max_i == 6:    
                # G7(x) =  x / (1 + x**2)   
                if fastica_on == True : 
                    grad = X.dot(((1 - yy)* inv_pabs_yy**2).T )/T
                    Edgx = np.sum(2 * y * (yy-3)* inv_pabs_yy* inv_pabs_yy* inv_pabs_yy)/T
                else :  
                    grad = X.dot((weight * (1 - yy) * inv_pabs_yy**2 ).T ) / np.sum(weight) 
                    vEGx = 2 * (EGx[6] > nf7['critical_point']) -1 
            if fastica_on == True :
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

        if fastica_on == True : 
            if cost_increaser_counter >= max_cost_increase_number  or 1- np.min(np.abs(np.diag(W.dot(last_W.T)))) < tolerance:   
                cost_increaser_counter = 0 
                W = np.copy(best_W ) 
                last_W = np.copy(W)
                iter_fastica = np.copy(iter)
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
    #if show_cost: 
        # insert plot here 
    ##############################################################################################################
    # Part 2: check for saddle points
    ##############################################################################################################
    if saddle_test_enable == True :
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
    

    if SADDLE_TESTED == True : 
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
        error = 0 

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
                    if fastica_on == True :  
                        grad = X.dot( (4* y* yy).T )/T 
                        Edgx = 12 
                    else : 
                        grad = X.dot((4 * weight * y * yy ).T ) / np.sum(weight)
                        vEGx = 2 * (EGx[0] > nf1['critical_point']) -1 
                elif max_i == 2:    
                    # G3(x) = np.abs(x)/ (1 + np.abs(x))    
                    if fastica_on == True : 
                        grad = X.dot( (sign_y * inv_pabs_y * inv_pabs_y).T )/T   
                        Edgx = np.sum(-2 * inv_pabs_y * inv_pabs_y * inv_pabs_y)/T 
                    else :
                        grad = X.dot((weight * sign_y * inv_pabs_y * inv_pabs_y).T ) / np.sum(weight) 
                        vEGx = 2 * (EGx[2] > nf3['critical_point']) -1 
                elif max_i == 4:    
                    # G5(x)  = x* np.abs(x) /(10 + np.abs(x))
                    if fastica_on == True : 
                        grad  = X.dot((abs_y *(20 + abs_y) * inv_p10abs_y * inv_p10abs_y).T )/T
                        Edgx = np.sum(200 * sign_y * inv_p10abs_y * inv_p10abs_y * inv_p10abs_y)/T  
                    else :
                        grad = X.dot((weight * abs_y * (20 + abs_y) * inv_p10abs_y**2 ).T ) / np.sum(weight) 
                        vEGx = 2 * (EGx[4] > nf5['critical_point']) -1  
                elif max_i == 6:    
                    # G7(x) =  x / (1 + x**2)   
                    if fastica_on == True : 
                        grad = X.dot(((1 - yy)* inv_pabs_yy**2).T )/T
                        Edgx = np.sum(2 * y * (yy-3)* inv_pabs_yy* inv_pabs_yy* inv_pabs_yy)/T
                    else :  
                        grad = X.dot((weight * (1 - yy) * inv_pabs_yy**2 ).T ) / np.sum(weight) 
                        vEGx = 2 * (EGx[6] > nf7['critical_point']) -1 
                if fastica_on == True :
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


            if fastica_on == True : 
                if cost_increaser_counter >= max_cost_increase_number  or 1- np.min(np.abs(np.diag(W.dot(last_W.T)))) < tolerance:   
                    cost_increaser_counter = 0 
                    W = np.copy(best_W ) 
                    last_W = np.copy(W)
                    iter_fastica = iter 
                    fastica_on = False 
                    continue
            else :  
                if cost_increaser_counter > stochastic_search_factor * max_cost_increase_number:
                    break
            last_W = np.copy(W)

    W = np.copy(best_W)   
    # if show cost:  to do later 

    # sort all components 
    max_negentropy, index_sort = np.sort(max_negentropy, axis = 0 )[::-1], np.argsort(max_negentropy, axis = 0)[::-1].flatten()
    W = W[index_sort, : ]    
##############################################################################################################
# Part 4: non-orthogonal ICA 
# fix small step size for refinement, gradient search 
##############################################################################################################
    if verbose:
        print('Non-orthogonal ICA stage.')    
    last_W = np.copy(W) 
    best_W = np.copy(W)
    Cost = np.zeros((max_iter_nonorth, 1)) 
    min_cost_queue = min_cost * np.ones((max_iter_nonorth, 1)) 
    error = np.inf
    mu = 1 / 25 
    min_mu = 1/ 200 
    max_cost_increase_number = 3 
    cost_increaser_counter = 0  
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
            grad = grad - ((w.T).dot(grad)) * w 
            grad = grad / np.linalg.norm(grad)  
            w1 = w + mu * grad
            w1 = w1 / np.linalg.norm(w1)   
            W[n, :] = np.copy(w1.T  )

        if Cost[iter] < min_cost:
            cost_increaser_counter = 0  
            min_cost = np.copy(Cost[iter])
            best_W = np.copy(last_W)
        else:
            cost_increaser_counter = cost_increaser_counter + 1
        
        min_cost_queue[iter] = np.copy(min_cost)
        if cost_increaser_counter > max_cost_increase_number:
            if mu > min_mu:
                cost_increaser_counter = 0  
                W = np.copy(best_W) 
                last_W = np.copy(W) 
                mu = mu/2   
                continue
            else:
                break   
        else:
            last_W = np.copy(W) 


    W = best_W
    W = W.dot(P)

    # if show cost:  to do later 

    return W 


###############################################################################################################
# These functions are used in the ICA-EBM algorithm.  
###############################################################################################################  


def simplified_ppval(pp, xs ):
    # simplified version of ppval 
    b = pp['breaks'][0]
    c = pp['coefs']
    l = int(pp['pieces'] ) 
    k = 4 
    dd = 1 
    # find index 
    index = float('nan ')
    middle_index = float('nan ')
    if xs > b[l-1]:
        index = l-1
    else:
        if xs < b[1]:
            index = 0
        else : 
            low_index = 0 
            high_index = l-1

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

def inv_sqrtmH(B):
    # inverse square root of a matrix   
    D, V = np.linalg.eig(B) 
    order = np.argsort(D) 
    D = D[order]
    V = V[:, order]  
    d = 1/np.sqrt(D) 
    A = np.dot(np.dot(V, np.diag(d)), V.T)  
    return A

def pre_processing(X):
    # pre-processing program
    N = X.shape[0]
    T = X.shape[1]
    # remove DC 
    Xmean = np.mean(X, axis=1) 
    X = X - np.tile(Xmean, (T, 1)).T
    # spatio pre-whitening
    R = np.dot(X, X.T) / T  
    P = inv_sqrtmH(R)
    X = np.dot(P, X)    
    return X, P

def symdecor(M): 
    # fast symmetric orthogonalization 
    D, V = np.linalg.eig(M.dot(M.T))    
    order = np.argsort(D)   
    D = D[order]
    V = V[:, order]    
    B = np.dot(np.ones((M.shape[1], 1)), np.reshape((1/np.sqrt(D)).T, (1, M.shape[1])   ))
    W = np.multiply(V, B ).dot(V.T.dot(M))
    return W

