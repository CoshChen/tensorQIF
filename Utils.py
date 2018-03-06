# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 13:04:18 2018

@author: Ko-Shin Chen
"""

import numpy as np

def get_M_list(T):
    M = []
    M.append(np.eye(T)) # M0 = id
    M.append(np.eye(T, k=1) + np.eye(T, k=-1)) # M1 = one upper/lower diagonal 1
    
    M2 = np.zeros([T,T])
    M2[0,0] = 1.0
    M2[T-1,T-1] = 1.0
    M.append(M2)
    
    M.append(np.ones([T,T]) - np.eye(T)) # M3 = all off-diagonal 1
    
    return M

def get_X_repeat(X, tau):
    '''
    @param X: numpy array [batch, T, d1, d2]
    @return: [batch, T-tau, (tau+1)*d1*d2]
    '''
    batch, T, d1, d2 = X.shape
    X_vect = np.reshape(X, [batch, T*d1*d2])
    
    X_repeat = np.zeros([batch, T-tau, (tau+1)*d1*d2])
    for t in range(T-tau):
        record_cols = [i for i in range(t*d1*d2, (t+tau+1)*d1*d2)]
        X_repeat[:, t, :] = X_vect[:, record_cols]
        
    return X_repeat
    
    


