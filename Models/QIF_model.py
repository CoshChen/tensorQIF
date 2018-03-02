# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 11:05:38 2018

@author: Ko-Shin Chen

This script runs experiments on the QIF model
and generates a CSV file that contains MSE for all 
datasets. The trained results and initial statess
are also saved in an npz file [W, initW]

The difference between QIF and Tensor QIF is that in
QIF, the current y_t only depends on the current X_t
"""

import numpy as np
import os
import csv
import QIF
import Utils

def minimize_QIF(X, y, M, L, tol, max_step, init_method, trained_record=None):
    '''
    @param X: numpy array [batch, T, d1, d2]
    @param y: numpy array [batch, T_y]
    @param tau: parameter in range(T)
    @param M: parameter numpy array list [M1, M2, ..., Md], 
              Mi are numpy arrays [T_y, T_y]; ref: Eq (8)
    @param L: one over the GD step size
    @param tol: termination condition
    @param max_step: the maximum number of iterations
    @param init_method: ways to initialize tensor coefficients;
                        option -- 'zeros' or 'random'
    @param trained_record: npz file contains [W, funvalue, initW]
    @return: [W, W_init], funvalue
             W are of dim [d1, d2]
             funvalue is a list of function values at each GD step
    '''
    
    if M[0].shape[0] != y.shape[1]:
        raise ValueError("M has incorrect dimensions.")
    
    X = X[:, :y.shape[1], :, :]
    # only use time points up to t=T_y <= T
    # Note that when tau = 0, the inputs 
    # X and y of tensorQIF_Gaussian(X, y, M, W)
    # should have the same number of time points.
    
    _, _, d1, d2 = X.shape
    
    
    if trained_record:
        npzfile = np.load(trained_record)
        W = np.expand_dims(npzfile['W'], axis=0)
        W_init = npzfile['initW']
    else:
        if init_method == 'zeros':
            for _ in range(3):
                W = np.zeros([1,d1,d2])
                W_init = W[0,:,:].copy() # same as np.squeeze(W)
        elif init_method == 'random':
            for _ in range(3):
                W = np.random.normal(size=(1,d1,d2), loc=0.0, scale=0.1)
                W_init = W[0,:,:].copy()
    
    fun_value = []       
    val, grad = QIF.tensorQIF_Gaussian(X, y, M, W)
    fun_value.append(val)
    
    print("Initial loss = " + str(val))
    print(" ")
    
    for i in range(max_step):
        W_new = W - (1.0/L)*grad
        val, grad = QIF.tensorQIF_Gaussian(X,y,M,W_new)
        
        # print("Step " + str(i) + ": " + str(val))
        if (i+1)%1000 == 0:
            print("Step " + str(i) + ": " + str(val))
        
        if abs(fun_value[-1] - val) < tol:
            fun_value.append(val)
            print("Early stop at step " + str(i))
            return [W_new[0,:,:], W_init], fun_value
        
        fun_value.append(val)
        W = W_new
        
    print("Done " + str(max_step) + "steps.")
    return [W[0,:,:], W_init], fun_value



'''
Training Script
'''
data_dir = '../SynData/500_5x5x5'
report_file = '../SynData/result/500_5x5x5/QIF_500_5x5x5.csv'
train_size = 400 # actual training size = train_size*(T-tau) 
dataset_num = 100

with open(report_file, 'w') as csvfile:
    fieldnames = ['QIF Training MSE','QIF Test MSE']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for data_id in range(dataset_num):
        print("Data #" + str(data_id))
        outfile_dir = '../SynData/result/500_5x5x5/500_5x5x5_' + str(data_id)
        trained_result =  outfile_dir + '/QIF_trainedW'
        
        if not os.path.exists(outfile_dir):
            os.makedirs(outfile_dir)
        
        data = data_dir + '/500_5x5x5_' + str(data_id) + '.npz'
        
        npzfile = np.load(data)
        X_train = npzfile['X'][:train_size, :,:,:]
        X_test = npzfile['X'][train_size:, :,:,:]
        y_train = npzfile['y'][:train_size, :]
        y_test = npzfile['y'][train_size:, :]
        
        M_list = Utils.get_M_list(y_train.shape[1])
        W_trained, vals = minimize_QIF(X_train, y_train, M_list[:2], 1, 10**-12, 100000, 'zeros')
        np.savez(trained_result, W=W_trained[0], initW = W_trained[1])
        
        # Calculate MSE
        _, T, d1, d2 = X_train.shape
        test_size = y_test.shape[0]
        
        X_train_flat = np.reshape(X_train, [train_size*T, d1*d2])
        X_test_flat = np.reshape(X_test, [test_size*T, d1*d2])
        W_vect = np.reshape(W_trained[0], [d1*d2, 1])
        
        y_pred_train = np.reshape(np.matmul(X_train_flat, W_vect), [train_size,T])[:, :y_train.shape[1]]
        y_pred_test = np.reshape(np.matmul(X_test_flat, W_vect), [test_size,T])[:, :y_test.shape[1]]
        train_MSE = np.mean(np.square(y_train - y_pred_train))
        test_MSE = np.mean(np.square(y_test - y_pred_test))
        
        # write in CSV file
        writer.writerow({'QIF Training MSE': train_MSE,
                         'QIF Test MSE': test_MSE})