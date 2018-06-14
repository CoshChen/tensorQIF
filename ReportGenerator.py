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
testset_num = 3
data_struct = '1000_3x2x2_AR08_c4'
test_data_struct = '10000_3x2x2_AR08_c4_test'
sample_size = 300
#################################

single_W = True
if model == 'tf_tensorQIF' or model == 'tensorQIF':
    single_W = False

data_dir = './SynData/' + data_struct
test_data_dir = './SynData/TestSet/' + str(test_data_struct)
result_dir = './SynData/result/' + str(data_struct) + '/m' + str(sample_size)
report_file = result_dir + '/' + model + '_' + data_struct + '.csv'

with open(report_file, 'w') as csvfile:
    fieldnames = ['Data #','Tr MSE','Va MSE'] + ['TePr MSE '+str(t) for t in range(testset_num)]+['TeTy MSE '+str(t) for t in range(testset_num)]+['W RMSE','Lambda']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for data_id in range(dataset_num):
        data = data_dir+'/'+data_struct+'_'+str(data_id)+'.npz'
        trained_result = result_dir +'/'+ data_struct +'_'+ str(data_id)+'/'+model+'_trainedW.npz'
        
        row = []
        
        if not os.path.exists(trained_result):
            print("No trained result for data # " + str(data_id))
            continue
        
        npzfile = np.load(data)
        
        cv_train_size = int(sample_size*0.8)
        X_train = npzfile['X'][:cv_train_size, :,:,:]
        X_cv = npzfile['X'][cv_train_size:sample_size, :,:,:]
        y_train = npzfile['y'][:cv_train_size, :]
        y_cv = npzfile['y'][cv_train_size:sample_size, :]
        
        W_true = npzfile['W1'] + npzfile['W2'] + npzfile['W3']
        lam = -1
        
        if single_W:
            W_trained = np.load(trained_result)['W']
        else:
            Ws = np.load(trained_result) # tensorQIF
            lam = Ws['lam']
            W_trained = Ws['W1'] + Ws['W2'] + Ws['W3']
        
        
        # Calculate MSE
        _, T, d1, d2 = X_train.shape
        cv_size = sample_size - cv_train_size
        
        if model in ['tensorQIF', 'tf_tensorQIF', 'LASSO_Granger']:
            tau = T - y_cv.shape[1]
        else:
            tau = 0
        
        X_train_flat = np.reshape(Utils.get_X_repeat(X_train,tau), [cv_train_size*(T-tau), (tau+1)*d1*d2])
        X_cv_flat = np.reshape(Utils.get_X_repeat(X_cv,tau), [cv_size*(T-tau), (tau+1)*d1*d2])
        W_vect = np.reshape(W_trained, [(tau+1)*d1*d2, 1])
        
        y_pred_train = np.reshape(np.matmul(X_train_flat, W_vect), [cv_train_size,T-tau])[:, :y_train.shape[1]]
        y_pred_cv = np.reshape(np.matmul(X_cv_flat, W_vect), [cv_size,T-tau])[:, :y_cv.shape[1]]
        train_MSE = np.mean(np.square(y_train - y_pred_train))
        vali_MSE = np.mean(np.square(y_cv - y_pred_cv))
        
        W_RMSE = np.sqrt(np.mean(np.square(W_true - W_trained)))
        
        row.append(data_id)
        row.append(train_MSE)
        row.append(vali_MSE)
        
        TePr = []
        TeTy = []
        for test_id in range(testset_num):
            test_file = np.load(test_data_dir+'/'+test_data_struct+'_'+str(test_id)+'.npz')
            
            X_test = test_file['X']
            y_test = test_file['y']
            W_true_vect = np.reshape(W_true, [(tau+1)*d1*d2, 1])
            test_size, _, _, _ = X_test.shape
            
            X_test_flat = np.reshape(Utils.get_X_repeat(X_test,tau), [test_size*(T-tau), (tau+1)*d1*d2])
            y_pred_test = np.reshape(np.matmul(X_test_flat, W_vect), [test_size,T-tau])[:, :y_test.shape[1]]
            
            TePr.append(np.mean(np.square(y_pred_test - y_test)))
            TeTy.append(np.mean(np.square(np.matmul(X_test_flat, W_vect - W_true_vect))))
        
        row += TePr
        row += TeTy
        
        row.append(W_RMSE)
        row.append(lam)
        
        # write in CSV file
        writer.writerow({fieldnames[col]: row[col] for col in range(len(fieldnames))})

