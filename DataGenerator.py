# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:14:39 2018

@author: Ko-Shin Chen


Generate synthetic data for the the case K=3 
For each sample, it contains a sequence of matrices 
M with dimension d1xd2:
[X(t=0), X(t=1), X(t=2), ... X(t=T-1)],
and a sequence (column) of scalars:
[y(t=0), y(t=1), y(t=2), ..., y(t=T-tau-1)].
Here t = 0 corresponds to the latest time point.

-----Output Format-----
X: observation [m, T, d1, d2]
y: labels [m, T-tau] 
W1: tensor coefficient [tau+1, d1, d2]
W2: tensor coefficient [tau+1, d1, d2]
W3: tensor coefficient [tau+1, d1, d2]

This scripts generates both .npz (for Python) and 
.mat (for MATLAB) data files.
"""

import os
import numpy as np
import scipy.io
import math
import Utils


############### Input Variables ###############
new_W = True
W_data = './SynData/1200_3x3x3_AR18_c4/1200_3x3x3_AR18_c4_0.npz' # path to assigned W

dataset_num = 100 # number of datasets needed
test_num = 1 # number of testsets
m = 2000 # sample size
test_m = 10000
# Sample size setting ref:
# https://stats.stackexchange.com/questions/60622/why-is-a-sample-covariance-matrix-singular-when-sample-size-is-less-than-number

T = 11 # number of time points
tau = 4
d1 = 10
d2 = 10

# None zero indicies (Patterns in Ws)
pattern_1 = [0] # time; numbers in range(tau)
pattern_2 = [0] # d1; numbers in range(d1)
pattern_3 = [0] #d2; numbers in range(d2)


cor = 'AR1'
alpha = 0.8 # Error term
c = 4 # var scale
###############################################


def get_X(m, T, d1, d2, mu, sigma):
    X = np.random.normal(mu, sigma, size=(m, T, d1, d2))
    
    for t in range(T):
        X[:, t, :, :] += np.random.uniform(low=0.0, high=np.sin(t)**2, size=(m,d1,d2))
    
    return X

def get_y(X, W, T, tau, alpha, c, cov_struct='AR1'):
    '''
    @param X: numpy array [m, T, d1, d2]
    @param W: numpy array [(tau+1), d1, d2]
    @param T: number of time points in dataset 
    @param tau: parameter in range(T-1)
    @param alpha: correlation parameter
    @return numpy array [m, T-tau]
    '''
    
    X_repeat = Utils.get_X_repeat(X, tau) # [m, T-tau, (tau+1)*d1*d2]
    
    _, d1, d2 = W.shape
    W_vect = np.reshape(W, [(tau+1)*d1*d2, 1])
    y = np.zeros([m, T-tau])
    
    mean = np.zeros(T-tau)
    cov = np.eye(T-tau)
    
    if cov_struct == 'AR1':
        for diff in range(1, T-tau):
            cov += (math.pow(alpha,diff)*np.eye(T-tau, k=diff))
            cov += (math.pow(alpha,diff)*np.eye(T-tau, k=-diff))
    
    elif cov_struct == 'exchangeable':
        cov += alpha*(np.ones([T-tau, T-tau]) - np.eye(T-tau))
    
    s = np.random.multivariate_normal(mean, c*cov, size = (m)) # [m, T-tau]
    
    for t in range(T-tau):
        y[:,t] = np.squeeze(np.matmul(X_repeat[:, t, :], W_vect))
    
    y += s
    # y += np.random.normal(loc=0.0, scale=0.05, size=y.shape)
    
    return y    



'''
Generate Datasets
'''
data_struct = str(m) +'_'+ str(tau+1)+'x'+str(d1)+'x'+str(d2)+'_'+cor+str(int(alpha*10))+'_c'+str(c)
data_dir = './SynData/' + data_struct
test_dir = data_dir + '/testset'
mat_dir = data_dir + '/matFiles'

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    
if not os.path.exists(test_dir):
    os.makedirs(test_dir)
    
if not os.path.exists(mat_dir):
    os.makedirs(mat_dir)

# Tensor Coefficients
if (not new_W) and W_data and os.path.exists(W_data):
    print("Use Assigned Ws.")
    d0 = np.load(W_data)
    W_list = [d0['W1'], d0['W2'], d0['W3']]

else:
    print("Generate new Ws.")
    W_list = []
    for _ in range(3):
        W_list.append(np.zeros([tau+1, d1, d2]))
        
    W_list[0][pattern_1, :, :] = 0.1*np.random.normal(0.0, 1.0, size=(len(pattern_1), d1, d2))
    W_list[1][:, pattern_2, :] = np.random.normal(0.0, 1.0, size=(tau+1, len(pattern_2), d2))
    W_list[2][:, :, pattern_3] = 0.01*np.random.normal(0.0, 1.0, size=(tau+1, d1, len(pattern_3)))
        
    # To generate very sparse coefficients
    #for _ in range(1):
    #    mask= np.random.randint(2, size=(tau+1, d1, d2))
    #    W_list[0] = np.multiply(W_list[0], mask)
    #    W_list[1] = np.multiply(W_list[1], mask)
    #    W_list[2] = np.multiply(W_list[2], mask)


W = W_list[0] + W_list[1] + W_list[2]

for data_id in range(dataset_num):
    outfile = data_dir + '/' + data_struct + '_'+ str(data_id) + '.npz'
    mat_file = mat_dir+'/'+data_struct+'_'+str(data_id) + '.mat'
    
    X = get_X(m, T, d1, d2, 0.0, 1.0) # [m, T, d1, d2]
    y = get_y(X, W, T, tau, alpha, c, cor) # [m, T-tau]
    
    np.savez(outfile, X=X, y=y, W1=W_list[0], W2=W_list[1], W3=W_list[2])
    scipy.io.savemat(mat_file, dict(X=X, y=y, W1=W_list[0], W2=W_list[1], W3=W_list[2]))
    
for test_id in range(test_num):
    outfile = test_dir + '/test_'+ str(test_id) + '.npz'
    mat_file = mat_dir+'/'+'test_'+str(test_id) + '.mat'
    
    X = get_X(m, T, d1, d2, 0.0, 1.0) # [m, T, d1, d2]
    y = get_y(X, W, T, tau, alpha, c, cor) # [m, T-tau]
    
    np.savez(outfile, X=X, y=y, W1=W_list[0], W2=W_list[1], W3=W_list[2])
    scipy.io.savemat(mat_file, dict(X=X, y=y, W1=W_list[0], W2=W_list[1], W3=W_list[2]))
    