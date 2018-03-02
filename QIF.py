# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 11:26:18 2018

@author: Ko-Shin Chen

TensorQIF Assumption:
The current y_t depends on current X_t and previous tau Xs

X_repeat = Utils.get_X_repeat(X, tau)
is used to reorganize the dataset to tensor records under this assumption.

For QIF model, tau=0
i.e. the current y_t depends only on current X_t
"""

import numpy as np
import Utils


def tensorQIF_Gaussian(X, y, M, W):
    '''
    Eq(4): mu = eta = <X, W>
    Eq(6): A = I
    
    @param X: numpy array [batch, T, d1, d2]
    @param y: numpy array [batch, T-tau]
    @param tau: parameter in range(T)
    @param M: parameter numpy array list [M1, M2, ..., Md], 
              Mi are numpy arrays [T-tau, T-tau]; ref: Eq (8)
    @param W: numpy [tau+1, d1, d2]; 
    '''
    
    batch, T, d1, d2 = X.shape # N = (tau+1)*d1*d2
    tau = T - y.shape[1]
    
    if tau < 0 or tau >= T:
        raise ValueError("Invalid tau. Check input dimensions.")
    
    if y.shape[1] != M[0].shape[0] or W.shape[0] != tau+1:
        raise ValueError("Incorrect Dimensions")
    
    d = len(M)
    
    X_repeat = Utils.get_X_repeat(X, tau) # same as D [batch, T-tau, (tau+1)*d1*d2]
    X_flat = np.reshape(X_repeat, [batch*(T-tau), (tau+1)*d1*d2])
    
    W_vect = np.reshape(W, [(tau+1)*d1*d2, 1])
    
    mu = np.reshape(np.matmul(X_flat, W_vect), [batch, T-tau])
    s = y - mu # [batch, T-tau]
    
    g_list = []
    dg_list = []
    
    for j in range(d):
        Ms = np.expand_dims(np.matmul(s, M[j]), axis=2) # [batch, T-tau, 1]
        D_tr = np.transpose(X_repeat, [0,2,1]) # [batch, (tau+1)*d1*d2, T-tau]
        g_list.append(np.matmul(D_tr, Ms)) # [batch, (tau+1)*d1*d2, 1]
        
        D_trM = np.zeros([batch, (tau+1)*d1*d2, T-tau])
        for b in range(batch):
            D_trM[b, :, :] = np.matmul(D_tr[b,:,:], M[j])
        
        dg_list.append(np.matmul(D_trM, X_repeat)) # [batch, (tau+1)*d1*d2, (tau+1)*d1*d2]
        
    g_i = np.concatenate(g_list, axis = 1) # [batch, d*(tau+1)*d1*d2, 1]
    dg_i = np.concatenate(dg_list, axis = 1) # [batch, d*(tau+1)*d1*d2, (tau+1)*d1*d2]
    
    g_m = np.mean(g_i, axis=0) # [d*(tau+1)*d1*d2, 1]
    g_m_tr = np.transpose(g_m)
    
    C_m = np.mean(np.matmul(g_i, np.transpose(g_i, [0,2,1])), axis=0)
    
    # cond = np.linalg.cond(C_m)
    # print("Current C_m's condition number = " + str(cond))
    
    C_m_inv = np.linalg.inv(C_m)
    # C_m_inv = np.linalg.pinv(C_m)
    
    dg_m = -np.mean(dg_i, axis=0) # [d*(tau+1)*d1*d2, (tau+1)*d1*d2]
    dg_m_tr = np.transpose(dg_m)
    
    # function value
    C_inv_g = np.matmul(C_m_inv, g_m)
    fun_value = batch * np.matmul(g_m_tr, C_inv_g)
    
    # function gradient
    '''
    J = np.matmul(dg_m_tr, np.matmul(C_m_inv, dg_m))
    J_inv = np.linalg.pinv(J)
    dQ_apx = 2.0*np.matmul(dg_m_tr, np.matmul(C_m_inv, g_m))
    fun_grad = np.matmul(J_inv, dQ_apx) # IRGLS direction
    '''
    
    dQ = 2.0*np.matmul(dg_m_tr, C_inv_g) # [(tau+1)*d1*d2, 1]
    g_tr_C_inv = np.matmul(g_m_tr, C_m_inv)
    
    for j in range((tau+1)*d1*d2):
        dC_i_part = np.matmul(np.expand_dims(dg_i[:,:,j], axis=2), np.transpose(g_i, [0,2,1])) # [batch, d*(tau+1)*d1*d2, d*(tau+1)*d1*d2]
        dC_m_part = np.mean(dC_i_part, axis=0)
        dC_entry = dC_m_part + np.transpose(dC_m_part) # the j-th entry of dC_m [d*(tau+1)*d1*d2, d*(tau+1)*d1*d2]
        dQ[j,0] = dQ[j,0] - np.matmul(np.matmul(g_tr_C_inv, dC_entry), C_inv_g)[0,0]
    
    fun_grad = dQ
    
    return np.squeeze(fun_value), np.reshape(fun_grad, [(tau+1),d1,d2])

