# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 19:54:31 2018

@author: Ko-Shin Chen

This script runs experiments on  
Granger model and generates a CSV file 
that contains MSE for all datasets. The trained 
results and initial states are also saved in 
an npz file [W]
"""

from sklearn import linear_model, metrics
import numpy as np
import os
import csv
import Utils

data_struct = '1000_3x2x2_AR08_c4'
test_data_struct = '10000_3x2x2_AR08_c4_test'
data_dir = '../SynData/' + data_struct
test_data_dir = '../SynData/TestSet/' + test_data_struct

train_size = 300 # actual training size = train_size*(T-tau)
result_dir = '../SynData/result/'+data_struct+'/m'+str(train_size)

tau = 2
report_file = result_dir+'/Granger_LASSO_'+data_struct+'_tau'+str(tau)+'test.csv'

dataset_num = 100
testset_num = 3



if not os.path.exists(result_dir):
    os.makedirs(result_dir)


with open(report_file, 'w') as csvfile:
    fieldnames = ['Data #','Tr MSE','Va MSE'] + ['TePr MSE '+str(t) for t in range(testset_num)]+['TeTy MSE '+str(t) for t in range(testset_num)]+['W RMSE','Lambda']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for data_id in range(dataset_num):
        data = data_dir + '/'+data_struct+'_' + str(data_id) + '.npz'
        outfile_dir = result_dir+'/'+data_struct+'_' + str(data_id)
        trained_result_LASSO = outfile_dir + '/LASSO_Granger_trainedW.npz'
        
        row = []
        
        if not os.path.exists(outfile_dir):
            os.makedirs(outfile_dir)
        
        npzfile = np.load(data)
        X = npzfile['X'][:train_size, :, :, :]
        y = npzfile['y'][:train_size, :]
        
        W_true = npzfile['W1'] + npzfile['W2'] + npzfile['W3']
        
        _, T, d1, d2 = X.shape
        true_tau = T - y.shape[1]
        
        Tx = min(T - (true_tau - tau), T)
        Ty = min(T - true_tau, T - tau)
        
        # Granger: use X_repeat instead of X
        X_vect = np.reshape(Utils.get_X_repeat(X[:,:Tx,:,:], tau), [train_size*Ty, (tau+1)*d1*d2])
        y_vect = np.reshape(y[:,:Ty], train_size*Ty)
        
        # LASSO
        # Choose Lambda
        scale = 0.001
        cv_train_size = int(train_size*0.8*Ty)
        final_lam = 0.0
        vali_MSE = 500
        
        for lam in range(1,51,2):
            gr_lasso = linear_model.Lasso(fit_intercept=False, alpha=lam*scale)
            gr_lasso.fit(X_vect[:cv_train_size, :], y_vect[:cv_train_size])
            gr_lasso_y_vali = gr_lasso.predict(X_vect[cv_train_size:, :])
            
            if metrics.mean_squared_error(y_vect[cv_train_size:], gr_lasso_y_vali) < vali_MSE:
                final_lam = lam*scale
                vali_MSE = metrics.mean_squared_error(y_vect[cv_train_size:], gr_lasso_y_vali)
        
        if final_lam == 0.0:
            final_lam = scale
        
        # Fit Lasso Granger for selected lambda
        gr_lasso = linear_model.Lasso(fit_intercept=False, alpha=final_lam)
        gr_lasso.fit(X_vect[:cv_train_size, :], y_vect[:cv_train_size])
        
        W_lasso = gr_lasso.coef_.reshape([(tau+1),d1,d2])
        np.savez(trained_result_LASSO, W=W_lasso)
        
        W_RMSE = np.sqrt(np.mean(np.square(W_true - W_lasso)))
        
        gr_lasso_y_pred_train = gr_lasso.predict(X_vect[:cv_train_size,:])
        gr_lasso_y_pred_cv = gr_lasso.predict(X_vect[cv_train_size:, :])
        train_MSE = metrics.mean_squared_error(y_vect[:cv_train_size], gr_lasso_y_pred_train)
        vali_MSE = metrics.mean_squared_error(y_vect[cv_train_size:], gr_lasso_y_pred_cv)
        
        row.append(data_id)
        row.append(train_MSE)
        row.append(vali_MSE)
        
        TePr = []
        TeTy = []
        for test_id in range(testset_num):
            test_file = np.load(test_data_dir+'/'+test_data_struct+'_'+str(test_id)+'.npz')
            X_test = test_file['X']
            test_size, _, _, _ = X_test.shape
            
            X_test_vect = np.reshape(Utils.get_X_repeat(X_test[:,:Tx,:,:], tau), [test_size*Ty, (tau+1)*d1*d2])
            y_test_vect = np.reshape(test_file['y'][:test_size, :Ty], test_size*Ty)
            
            # compute predicting MSE
            gr_lasso_y_pred_test = gr_lasso.predict(X_test_vect)
            TePr.append(metrics.mean_squared_error(y_test_vect, gr_lasso_y_pred_test))
            
            # compute true y MSE
            TeTy.append(np.mean(np.square(np.matmul(X_test_vect, np.reshape(W_true - W_lasso, [(tau+1)*d1*d2, 1])))))
        
        row += TePr
        row += TeTy
        
        row.append(W_RMSE)
        row.append(final_lam)
        
        # write in CSV file
        writer.writerow({fieldnames[col]: row[col] for col in range(len(fieldnames))})


