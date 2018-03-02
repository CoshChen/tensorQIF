# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 13:04:18 2018

@author: Ko-Shin Chen
"""

import numpy as np

def get_M_list(T):
    M = []
    M.append(np.eye(T)) # id
    M.append(np.ones([T,T]) - np.eye(T)) # all off-diagonal 1
    M.append(np.eye(T, k=1) + np.eye(T, k=-1)) # one upper/lower diagonal 1
    
    M4 = np.zeros([T,T])
    M4[0,0] = 1.0
    M4[T-1,T-1] = 1.0
    
    M.append(M4)
    
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
    
    


