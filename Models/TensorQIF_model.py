# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 16:45:38 2018

@author: Ko-Shin Chen


"""

import numpy as np
import QIF
import Norms


def minimize_tensorQIF(X, y, tau, M, lam_list, L, tol, max_step, init_method):
    '''
    @param X: numpy array [batch, T, d1, d2]
    @param y: numpy array [batch, T_y]
    @param tau: parameter in range(T)
    @param M: parameter numpy array list [M1, M2, ..., Md], 
              Mi are numpy arrays [T-tau, T-tau]; ref: Eq (8)
    @param lam_list: [lambda_1, lambda_2, lambda_3]
    @param L: one over the GD step size
    @param tol: termination condition
    @param max_step: the maximum number of iterations
    @param init_method: ways to initialize tensor coefficients;
                        option -- 'zeros' or 'random'
    @return: [W1, W2, W3, W1_init, W2_init, W3_init], funvalue
             Wi are of dim [(tau+1), d1, d2]
             funvalue is a list of function values at each GD step
    '''
    
    _, T, d1, d2 = X.shape
    
    if y.shape[1] < T-tau:
        raise ValueError("The label -- y has no enough records.")
    
    if M[0].shape[0] != T-tau:
        raise ValueError("M has incorrect dimensions.")
    
    y = y[:, :(T-tau)]
    
    W_list = []
    W_init = []
    
    if init_method == 'zeros':
        for _ in range(3):
            W_list.append(np.zeros([(tau+1),d1,d2]))
            W_init.append(W_list[-1].copy())
    elif init_method == 'random':
        for _ in range(3):
            W_list.append(np.random.normal(size=((tau+1),d1,d2), loc=0.0, scale=0.1))
            W_init.append(W_list[-1].copy())
            
    
    fun_value = []
    
    W = W_list[0] + W_list[1] + W_list[2]
    val, grad = QIF.tensorQIF_Gaussian(X, y, M, W)
    val += Norms.latent_LF1(lam_list, W_list)
    fun_value.append(val)
    
    print("Initial loss = " + str(val))
    print(" ")
    
    for i in range(max_step):
        W_new_list = []
        for k in range(3):
            W_new_list.append(_update_W(k, W_list[k], grad, lam_list[k], L, tau, d1, d2))
        
        print("---Step " + str(i) + "---")
        W_new = W_new_list[0] + W_new_list[1] + W_new_list[2]
        val, grad = QIF.tensorQIF_Gaussian(X, y, M, W_new)
        val += Norms.latent_LF1(lam_list, W_new_list)
        
        print("Loss = " + str(val))
        
        if abs(fun_value[-1] - val) < tol:
            fun_value.append(val)
            print("Early stop at step " + str(i))
            return W_new_list + W_init, fun_value
        
        elif fun_value[-1] < val:
            print("Early stop at step " + str(i))
            return W_list + W_init, fun_value
        
        fun_value.append(val)
        W_list = W_new_list
        print(" ")
        
    print("Done " + str(max_step) + "steps.")
    return W_list + W_init, fun_value


def _update_W(k, W, grad, lam, L, tau, d1, d2):
    L *= 3.0
    P = W - grad/L
    
    W_new = np.zeros([tau+1, d1, d2])
    dim = W.shape[k]
    
    if k == 0:
        for i in range(dim):
            a = 1.0 - lam/(L*Norms.L2(P[i,:,:]))
            if a > 0: W_new[i,:,:] = a*P[i,:,:]
    elif k == 1:
        for i in range(dim):
            a = 1.0 - lam/(L*Norms.L2(P[:,i,:]))
            if a > 0: W_new[:,i,:] = a*P[:,i,:]
    else:
        for i in range(dim):
            a = 1.0 - lam/(L*Norms.L2(P[:,:,i]))
            if a > 0: W_new[:,:,i] = a*P[:,:,i]
            
    return W_new