# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 10:15:26 2018

@author: Ko-Shin Chen
"""

import tensorflow as tf
import math

def QIF_Gaussian_tf(X_repeat, y, M, W):
    '''
    Eq(4): mu = eta = <X, W>
    Eq(6): A = I
    
    @param X_repeat: placeholder [batch, T-tau, (tau+1)*d1*d2]
    @param y: placeholder [batch, T-tau]
    @param M: placeholder list [M1, M2, ..., Md], 
              Mi are of dim [T-tau, T-tau]; ref: Eq (8)
    @param W: [tau+1, d1, d2]; 
    '''
    batch = tf.shape(X_repeat)[0]
    dims = X_repeat.get_shape() # [batch, T-tau, (tau+1)*d1*d2]
    
    W_vect = tf.reshape(W, [int(dims[2]), 1]) # [(tau+1)*d1*d2, 1]
    X_flat = tf.reshape(X_repeat, [batch*int(dims[1]), int(dims[2])]) # [batch*(T-tau), (tau+1)*d1*d2]
    mu = tf.reshape(tf.matmul(X_flat, W_vect), [batch, int(dims[1])], name='mu') # [batch, T-tau]
    s = tf.subtract(y, mu, name='s')
    
    d = len(M)
    g_list = []
    
    for j in range(d):
        Ms = tf.expand_dims(tf.matmul(s, M[j]), -1) # [batch, T-tau, 1]
        D_tr_Ms = tf.matmul(tf.transpose(X_repeat, [0,2,1]), Ms, name='D_tr_Ms_'+str(j)) # [batch, (tau+1)*d1*d2, 1]
        g_list.append(D_tr_Ms)
        
    g_i = tf.concat(g_list, 1, name='g_i') # [batch, d*(tau+1)*d1*d2, 1]
    g_m = tf.reduce_mean(g_i, 0, name='g_m') # [d*(tau+1)*d1*d2, 1]
    g_m_tr = tf.transpose(g_m)
    
    C_i = tf.matmul(g_i, tf.transpose(g_i,[0,2,1]), name='C_i') # [batch, d*(tau+1)*d1*d2, d*(tau+1)*d1*d2]
    C_m = tf.reduce_mean(C_i, 0, name='C_m') # [d*(tau+1)*d1*d2, d*(tau+1)*d1*d2]
    
    C_m_inv = tf.matrix_inverse(C_m, name='C_m_inv')
    
    val = tf.matmul(g_m_tr, tf.matmul(C_m_inv, g_m))
    
    return tf.multiply(tf.squeeze(val), tf.cast(batch, dtype='float64'), name='QIF')
    

def tensor_L12(lam, W_slices, name='tensor_L12'):
    lam *= math.sqrt(2.0)
    norms = [tf.sqrt(tf.nn.l2_loss(W_slices[0]))]
    for i in range(1, len(W_slices)):
        norms.append(tf.add(norms[i-1], tf.sqrt(tf.nn.l2_loss(W_slices[i]))))
            
    return tf.multiply(norms[-1], lam, name=name)
            
