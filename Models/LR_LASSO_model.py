# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:44:58 2018

@author: Ko-Shin Chen

This script runs experiments on LR and LASSO models
and generates a CSV file that contains MSE for all 
datasets.
"""

from sklearn import linear_model, metrics
import numpy as np
import csv

data_struct = '2000_5x5x5'
data_dir = '../SynData/'+data_struct
report_file = '../SynData/result/'+data_struct+'/LR_LASSO_'+data_struct+'.csv'
train_size = 1500 # actual training size = train_size*(T-tau) 
dataset_num = 100

with open(report_file, 'w') as csvfile:
    fieldnames = ['Data #', 'LR Training MSE','LR Test MSE', 'LASSO Training MSE', 'LASSO Test MSE']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for data_id in range(dataset_num):
        data = data_dir +'/'+data_struct+'_' + str(data_id) + '.npz'
        
        npzfile = np.load(data)
        X = npzfile['X']
        y = npzfile['y']
        batch, T, d1, d2 = X.shape
        tau = T - y.shape[1]
        
        X_vect = np.zeros([batch*(T-tau), d1*d2])
        for b in range(batch):
            for t in range(T-tau):
                X_vect[b*(T-tau) + t,:] = np.reshape(X[b,t,:,:], d1*d2)
        
        y_vect = np.reshape(y, batch*(T-tau))
        
        # Training and test sets
        X_train = X_vect[:train_size*(T-tau), :]
        X_test = X_vect[train_size*(T-tau):, :]
        y_train = y_vect[:train_size*(T-tau)]
        y_test = y_vect[train_size*(T-tau):]
        
        # Linear Regresssion
        lr = linear_model.LinearRegression()
        lr.fit(X_train, y_train)
        lr_y_pred_train = lr.predict(X_train)
        lr_y_pred_test = lr.predict(X_test)
        lr_train_MSE = metrics.mean_squared_error(y_train, lr_y_pred_train)
        lr_test_MSE = metrics.mean_squared_error(y_test, lr_y_pred_test)
        
        # LASSO
        lasso = linear_model.Lasso(alpha=1.0)
        lasso.fit(X_train, y_train)
        lasso_y_pred_train = lasso.predict(X_train)
        lasso_y_pred_test = lasso.predict(X_test)
        lasso_train_MSE = metrics.mean_squared_error(y_train, lasso_y_pred_train)
        lasso_test_MSE = metrics.mean_squared_error(y_test, lasso_y_pred_test)
        
        # write in CSV file
        writer.writerow({'Data #': data_id,
                         'LR Training MSE': lr_train_MSE,
                         'LR Test MSE': lr_test_MSE, 
                         'LASSO Training MSE': lasso_train_MSE,
                         'LASSO Test MSE': lasso_test_MSE})