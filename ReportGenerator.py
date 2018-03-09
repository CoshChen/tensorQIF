# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 15:28:05 2018

@author: Ko-Shin Chen
"""

import csv
import numpy as np
import os
import Utils


######## Input Variables ########
model = 'tf_tensorQIF'
dataset_num = 100
data_struct = '500_5x5x5'
train_size = 400
#################################

single_W = True
if model == 'tf_tensorQIF' or model == 'tensorQIF':
    single_W = False

data_dir = './SynData/' + data_struct
result_dir = './SynData/result/' + str(data_struct)
report_file = result_dir + '/' + model + '_' + data_struct + '.csv'

with open(report_file, 'w') as csvfile:
    fieldnames = ['Data #', model + ' Training MSE', model + ' Test MSE']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for data_id in range(dataset_num):
        data = data_dir+'/'+data_struct+'_'+str(data_id)+'.npz'
        trained_result = result_dir +'/'+ data_struct +'_'+ str(data_id)+'/'+model+'_trainedW.npz'
        
        if not os.path.exists(trained_result):
            print("No trained result for data # " + str(data_id))
            continue
        
        npzfile = np.load(data)
        X_train = npzfile['X'][:train_size, :,:,:]
        X_test = npzfile['X'][train_size:, :,:,:]
        y_train = npzfile['y'][:train_size, :]
        y_test = npzfile['y'][train_size:, :]
        
        if single_W:
            W_trained = np.load(trained_result)['W']
        else:
            Ws = np.load(trained_result) # tensorQIF
            W_trained = Ws['W1'] + Ws['W2'] + Ws['W3']
        
        
        # Calculate MSE
        _, T, d1, d2 = X_train.shape
        test_size = y_test.shape[0]
        
        if model in ['tensorQIF', 'tf_tensorQIF', 'Granger', 'LASSO_Granger']:
            tau = T - y_test.shape[1]
        else:
            tau = 0
        
        X_train_flat = np.reshape(Utils.get_X_repeat(X_train,tau), [train_size*(T-tau), (tau+1)*d1*d2])
        X_test_flat = np.reshape(Utils.get_X_repeat(X_test,tau), [test_size*(T-tau), (tau+1)*d1*d2])
        W_vect = np.reshape(W_trained, [(tau+1)*d1*d2, 1])
        
        y_pred_train = np.reshape(np.matmul(X_train_flat, W_vect), [train_size,T-tau])[:, :y_train.shape[1]]
        y_pred_test = np.reshape(np.matmul(X_test_flat, W_vect), [test_size,T-tau])[:, :y_test.shape[1]]
        train_MSE = np.mean(np.square(y_train - y_pred_train))
        test_MSE = np.mean(np.square(y_test - y_pred_test))
        
        # write in CSV file
        writer.writerow({'Data #': data_id, 
                         model + ' Training MSE': train_MSE,
                         model + ' Test MSE': test_MSE})

