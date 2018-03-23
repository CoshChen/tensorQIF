# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 19:54:31 2018

@author: Ko-Shin Chen

This script runs experiments on Granger and 
LASSO Granger models and generates a CSV file 
that contains MSE for all datasets. The trained 
results and initial states are also saved in 
an npz file [W]
"""

from sklearn import linear_model, metrics
import numpy as np
import os
import csv
import Utils

data_struct = '1000_5x5x5_Exch08'
data_dir = '../SynData/' + data_struct
result_dir = '../SynData/result/'+data_struct
report_file = result_dir+'/Granger_LASSO_'+data_struct+'.csv'
train_size = 800 # actual training size = train_size*(T-tau)
dataset_num = 100

if not os.path.exists(result_dir):
    os.makedirs(result_dir)


with open(report_file, 'w') as csvfile:
    fieldnames = ['Data #', 'Granger Training MSE', 'Granger Test MSE', 'LASSO Granger Training MSE','LASSO Granger Test MSE', 'Lambda']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for data_id in range(dataset_num):
        data = data_dir + '/'+data_struct+'_' + str(data_id) + '.npz'
        outfile_dir = result_dir+'/'+data_struct+'_' + str(data_id)
        trained_result = outfile_dir + '/Granger_trainedW.npz'
        trained_result_LASSO = outfile_dir + '/LASSO_Granger_trainedW.npz'
        
        if not os.path.exists(outfile_dir):
            os.makedirs(outfile_dir)
        
        npzfile = np.load(data)
        X = npzfile['X']
        y = npzfile['y']
        batch, T, d1, d2 = X.shape
        tau = T - y.shape[1]
        
        # Granger: use X_repeat instead of X
        X_vect = np.reshape(Utils.get_X_repeat(X, tau), [batch*(T-tau), (tau+1)*d1*d2])
        y_vect = np.reshape(y, batch*(T-tau))
        
        # Training and test sets
        X_train = X_vect[:train_size*(T-tau), :]
        X_test = X_vect[train_size*(T-tau):, :]
        y_train = y_vect[:train_size*(T-tau)]
        y_test = y_vect[train_size*(T-tau):]
        
        # Linear Regresssion
        gr = linear_model.LinearRegression(fit_intercept=False)
        gr.fit(X_train, y_train)
        
        W = gr.coef_.reshape([(tau+1),d1,d2])
        np.savez(trained_result, W=W)
        
        gr_y_pred_train = gr.predict(X_train)
        gr_y_pred_test = gr.predict(X_test)
        gr_train_MSE = metrics.mean_squared_error(y_train, gr_y_pred_train)
        gr_test_MSE = metrics.mean_squared_error(y_test, gr_y_pred_test)       
        
        # LASSO
        # Choose Lambda
        scale = 0.01
        cv_train_size = int((train_size/2)*(T-tau))
        final_lam = 0.0
        vali_MSE = 2*gr_test_MSE
        
        for lam in range(1,6):
            gr_lasso = linear_model.Lasso(fit_intercept=False, alpha=lam*scale)
            gr_lasso.fit(X_train[:cv_train_size, :], y_train[:cv_train_size])
            gr_lasso_y_vali = gr_lasso.predict(X_train[cv_train_size:, :])
            
            if metrics.mean_squared_error(y_train[cv_train_size:], gr_lasso_y_vali) < vali_MSE:
                final_lam = lam*scale
                vali_MSE = metrics.mean_squared_error(y_train[cv_train_size:], gr_lasso_y_vali)
        
        # Fit Lasso Granger for all training data
        gr_lasso = linear_model.Lasso(fit_intercept=False, alpha=final_lam)
        gr_lasso.fit(X_train, y_train)
        
        W_lasso = gr_lasso.coef_.reshape([(tau+1),d1,d2])
        np.savez(trained_result_LASSO, W=W_lasso)
        
        gr_lasso_y_pred_train = gr_lasso.predict(X_train)
        gr_lasso_y_pred_test = gr_lasso.predict(X_test)
        gr_lasso_train_MSE = metrics.mean_squared_error(y_train, gr_lasso_y_pred_train)
        gr_lasso_test_MSE = metrics.mean_squared_error(y_test, gr_lasso_y_pred_test)
        
        # write in CSV file
        writer.writerow({'Data #': data_id,
                         'Granger Training MSE': gr_train_MSE, 
                         'Granger Test MSE': gr_test_MSE,
                         'LASSO Granger Training MSE': gr_lasso_train_MSE,
                         'LASSO Granger Test MSE': gr_lasso_test_MSE,
                         'Lambda': final_lam})


