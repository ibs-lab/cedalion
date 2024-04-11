import scipy as sp
import numpy as np
import matplotlib.pyplot as plt 
import ICA_EBM as ICA_EBM 

def ERBM(X, p = np.nan ):
    # BSS, originally introduced  as full BSS
    # Reference:
    # This code is based on the matlab version of bss by Xi-Lin Li 
    # 
    # Xi-Lin Li, Tulay Adali, "Blind spatiotemporal separation of second and/or 
    # higher-order correlated sources by entropy rate minimization," 
    # IEEE International Conference on Acoustics, Speech and Signal Processing 2010. 

    # Inputs:  
    # X: mixtures 
    # p: filter length 
    #
    # Outputs:
    # W: estimated demixing matrix

#################  Part 0: pre-processing #################

    # load measuring functions as global variables 
    global nf1, nf2, nf3, nf4, nf5, nf6, nf7, nf8   
    table = np.load('measfunc_table.npy', allow_pickle= True)
    K = 8 
    nf1, nf2, nf3, nf4, nf5, nf6, nf7, nf8 = table[0], table[1], table[2], table[3], table[4], table[5], table[6], table[7]
   
    show_cost = False 
    N, T = X.shape 
    X, P = pre_processing(X)

    # initialize p if it is not provided    
    if np.isnan(p): 
        p = np.min(11, T/ 50)

    # initialize W
    W = ICA_EBM.ICA_EBM(X) 

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

    # prepare the data used in integral 
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
        W = np.copy(best_W)
        a = np.copy(best_a) 
        last_W = np.copy(best_W)
        Cost = np.zeros((max_iter_north, 1))    
        min_cost = np.inf   
        min_cost_queue = min_cost * np.ones((max_iter_north, 1))
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

                # constant direction 
                cnstd = Rz[:, :, n].dot(v) 
                # projected gradient    
                grad =  grad - (cnstd.T.dot(grad) * cnstd /(np.dot(cnstd.T, cnstd))).reshape(-1, 1) 
                grad = inv_sqrtmH(Rz[:, :, n]).dot(grad)
                # normalized gradient
                grad = grad / np.sqrt(grad.T.dot(Rz[:, :, n].dot(grad)))
                v = v.reshape(-1,1) + mu * grad      
                v = v / np.sqrt(v.T.dot(Rz[:, :, n].dot(v)))   
                W[n, :] = np.copy(v.T )   
              
            
            if Cost[iter-1]  < min_cost:
                cost_increase_counter = 0
                min_cost = np.copy(Cost[iter-1])
                best_W = np.copy(last_W)
                best_a = np.copy(a)
                max_negentropy = np.copy(negentropy_array)   
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
                        mu = mu / 2 
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
                        mu = mu / 2 
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


def lfc(x, p, choice, a0): 
    # return the linear filtering coefficients (LFC) with length p for entropy 
    # rate estimation, and the estimated entropy rate
    # 
    # Inputs: 
    # p: is the filter length
    # 'choice':  can be 'sub', 'super', or 'unknown'
    # a0:  is the intial guess
    # 
    # Outputs
    # a:  is the filter coefficients
    # min_cost: is the entropy rate estimation
    global nf1, nf2, nf3, nf4, nf5, nf6, nf7, nf8   
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

def cnstd_and_gain(a):
    # return constraint direction used for calculating 
    # projected gradient and Gain of filter a
    global cosmtx, sinmtx, Simpson_c
    eps = np.finfo(np.float64).eps  
    p = a.shape[0]  
    # calculate the integral 
    # sample omega from 0 to pi 
    n = 10*p    
    h = np.pi / n   
    w = np.arange(0, n+1, 1) * h    

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
    b = sp.linalg.toeplitz(u[:p]).dot(a)

    # gain 
    G = u[p] 
    return b, G 

 

def calculate_cos_sin_mtx(p): 
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


def pre_processing(X):
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

    af  = np.linalg.solve(sp.linalg.toeplitz(r[:q-1]), np.conjugate(r[1:]) )
    for n in range(N): 
        X[n, :] =  sp.signal.lfilter(np.concatenate((np.ones((1,1)), -af), axis = 0)[:,0], 1 ,X[n, :])

    # spatio pre-whitening
    R = np.dot(X, X.T) / T
    P2 = inv_sqrtmH(R)
    X = np.dot(P2, X)
    P = np.dot(P2, P1)

    return X, P 

def inv_sqrtmH(B):
    # calculate the inverse square root 
    D, V = np.linalg.eig(B) 
    order = np.argsort(D) 
    D = D[order]
    V = V[:, order]  
    d = 1/np.sqrt(D) 
    A = np.dot(np.dot(V, np.diag(d)), V.T)  
    return A

 