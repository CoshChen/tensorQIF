# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 10:37:56 2018

@author: Ko-Shin Chen
"""

import numpy as np

def latent_LF1(lam_list, W_list):
    norm = 0.0
    
    for k in range(3):
        norm += (lam_list[k] * _LF1(k, W_list[k]))     
    return norm
    

def _LF1(k, tensor):
    dim = tensor.shape[k]
    norm = 0.0
    
    if k == 0:
        for i in range(dim):
            norm += L2(tensor[i,:,:])
    elif k == 1:
        for i in range(dim):
            norm += L2(tensor[:,i,:])
    else:
        for i in range(dim):
            norm += L2(tensor[:,:,i])
    return norm    


def L2(array):
    return np.sqrt(np.sum(np.square(array)))

def L2_squared(array):
    return np.sum(np.square(array))

def L1(matrix):
    return max(np.sum(np.abs(matrix),axis=0))

def L_inf(matrix):
    return max(np.sum(np.abs(matrix),axis=1))