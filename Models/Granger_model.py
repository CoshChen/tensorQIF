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
import csv
import Utils

data_dir = '../SynData/500_5x5x5'
outfile_dir = '../SynData/result/500_5x5x5'
report_file = outfile_dir + '/Granger_LASSO_500_5x5x5.csv'
train_size = 400 # actual training size = train_size*(T-tau) 
dataset_num = 100

with open(report_file, 'w') as csvfile:
    fieldnames = ['Data #', 'Granger Training MSE', 'Granger Test MSE', 'LASSO Granger Training MSE','LASSO Granger Test MSE']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for data_id in range(dataset_num):
        data = data_dir + '/500_5x5x5_' + str(data_id) + '.npz'
        trained_result = outfile_dir + '/500_5x5x5_' + str(data_id) + '/Granger_trainedW.npz'
        trained_result_LASSO = outfile_dir + '/500_5x5x5_' + str(data_id) + '/LASSO_Granger_trainedW.npz'
        
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
        gr = linear_model.LinearRegression()
        gr.fit(X_train, y_train)
        
        W = gr.coef_.reshape([(tau+1),d1,d2])
        np.savez(trained_result, W=W)
        
        gr_y_pred_train = gr.predict(X_train)
        gr_y_pred_test = gr.predict(X_test)
        gr_train_MSE = metrics.mean_squared_error(y_train, gr_y_pred_train)
        gr_test_MSE = metrics.mean_squared_error(y_test, gr_y_pred_test)       
        
        # LASSO
        gr_lasso = linear_model.Lasso(alpha=1.0)
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
                         'LASSO Granger Test MSE': gr_lasso_test_MSE})


