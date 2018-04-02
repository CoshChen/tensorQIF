# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 16:45:38 2018

@author: Ko-Shin Chen


"""

import numpy as np
import matplotlib.pyplot as plt
import os
import QIF
import Norms
import Utils


def minimize_tensorQIF(X, y, tau, M, lam_list,
                       line_search,
                       tol, max_step, 
                       init_method,
                       corr_restruct=False, alpha=10**-3,
                       trained_record=None):
    '''
    @param X: numpy array [batch, T, d1, d2]
    @param y: numpy array [batch, T_y]
    @param tau: parameter in range(T)
    @param M: parameter numpy array list [M1, M2, ..., Md], 
              Mi are numpy arrays [T-tau, T-tau]; ref: Eq (8)
    @param lam_list: [lambda_1, lambda_2, lambda_3]
    @param line_search: True for using line search
    @param tol: termination condition
    @param max_step: the maximum number of iterations
    @param init_method: ways to initialize tensor coefficients;
                        option -- 'zeros' or 'path to assigned init file'
    @param corr_restruct: set to True if use C_m = (1-alpha)Sample_Corr + alpha*I
    @alpha: C_m = I if alpha >= 1.0
    @return: [W1, W2, W3, W1_init, W2_init, W3_init], funvalue
             Wi are of dim [(tau+1), d1, d2]
             funvalue is a list of function values at each GD step
    '''
    
    _, T, d1, d2 = X.shape
    
    if y.shape[1] < T-tau:
        raise ValueError("The label -- y has no enough records.")
    
    if M[0].shape[0] != T-tau:
        raise ValueError("M has incorrect dimensions.")
    
    y = y[:, :(T-tau)]
    
    W_list = []
    W_init = []
    
    if trained_record and os.path.exists(trained_record):
        print("Load Trained File")
        npzfile = np.load(trained_record)
        W_list.append(npzfile['W1'])
        W_list.append(npzfile['W2'])
        W_list.append(npzfile['W3'])
        
        W_init.append(npzfile['W1_init'])
        W_init.append(npzfile['W2_init'])
        W_init.append(npzfile['W3_init'])
        
    elif os.path.exists(init_method):
        print("Use Assigned Initial Values")
        npzfile = np.load(init_method)
        err = np.random.uniform(low=-0.1, high=0.1, size=npzfile['W1'].shape)
        W_list.append(npzfile['W1']+err)
        W_list.append(npzfile['W2']+err)
        W_list.append(npzfile['W3']+err)
        
        W_init.append(W_list[0].copy())
        W_init.append(W_list[1].copy())
        W_init.append(W_list[2].copy())
                
    elif init_method == 'zeros':
        print("Use Zeros as Initial Values")
        for _ in range(3):
            W_list.append(np.zeros([(tau+1),d1,d2]))
            W_init.append(W_list[-1].copy())
    else:
        print("Random Initial Values")
        for _ in range(3):
            W_list.append(np.random.normal(size=((tau+1),d1,d2), loc=0.0, scale=0.1))
            W_init.append(W_list[-1].copy())
            
    
    fun_value = []
    
    W = W_list[0] + W_list[1] + W_list[2]
    
    if line_search:
        L = 1.0
        val, grad, _ = QIF.tensorQIF_Gaussian(X, y, M, W, corr_restruct, alpha)
    else:
        val, grad, L = QIF.tensorQIF_Gaussian(X, y, M, W, corr_restruct, alpha, True) # Calculate L_max
        print("L = " + str(L))
        
    fun_value.append(val + Norms.latent_LF1(lam_list, W_list))
    
    print("Initial loss = " + str(fun_value[-1]))
    print(" ")
    
    for i in range(max_step):
        if line_search:
            # ref: http://www.seas.ucla.edu/~vandenbe/236C/lectures/proxgrad.pdf page:6-20
            max_iter = 10**4
            for line_iter in range(max_iter):
                W_new_list = []
                for k in range(3):
                    W_new_list.append(_update_W(k, W_list[k], grad, lam_list[k], L, tau, d1, d2))
                
                W_new = W_new_list[0] + W_new_list[1] + W_new_list[2]
                new_val, _, _ = QIF.tensorQIF_Gaussian(X, y, M, W_new, corr_restruct, alpha)
                LH = new_val - val + np.sum(np.multiply(grad, W - W_new))
                
                if 2*LH <= L*Norms.L2_squared(W - W_new) or line_iter == max_iter-1:
                    W_list = W_new_list
                    W = W_new
                    val, grad, _ = QIF.tensorQIF_Gaussian(X, y, M, W, corr_restruct, alpha)
                    break
                else:
                    L *= 2.0
                
        else:
            W_new_list = []
            for k in range(3):
                W_new_list.append(_update_W(k, W_list[k], grad, lam_list[k], L, tau, d1, d2))
                
            W_list = W_new_list
            W = W_new_list[0] + W_new_list[1] + W_new_list[2]
            val, grad, _ = QIF.tensorQIF_Gaussian(X, y, M, W, corr_restruct, alpha) 
                
        fun_value.append(val + Norms.latent_LF1(lam_list, W_list))
        
        if (i+1)%500 == 0:
            print("---Step " + str(i) + "---")
            print("Loss = " + str(fun_value[-1]))
            print(" ")
        
        if abs(fun_value[-1] - fun_value[-2]) < tol:
            print("Early stop at step " + str(i))
            return W_list + W_init, fun_value
        
    print("Done " + str(max_step) + "steps.")
    return W_list + W_init, fun_value


def _update_W(k, W, grad, lam, L, tau, d1, d2):
    L *= 3.0
    P = W - grad/L
    
    W_new = np.zeros([tau+1, d1, d2])
    dim = W.shape[k]
    
    if k == 0:
        for i in range(dim):
            a = 1.0 - lam/(L*Norms.L2(P[i,:,:]))
            if a > 0: W_new[i,:,:] = a*P[i,:,:]
    elif k == 1:
        for i in range(dim):
            a = 1.0 - lam/(L*Norms.L2(P[:,i,:]))
            if a > 0: W_new[:,i,:] = a*P[:,i,:]
    else:
        for i in range(dim):
            a = 1.0 - lam/(L*Norms.L2(P[:,:,i]))
            if a > 0: W_new[:,:,i] = a*P[:,:,i]
            
    return W_new



'''
Training Script (Single)
'''
data_struct = '150_10x10x10_AR08'
data_dir = '../SynData/' + data_struct
train_size = 120
#dataset_num = 100

lam_list = [10,10,10]
tol = 10**-12
max_step = 10**5
M_list_len = 2

line_search = False
corr_restruct = False
alpha = 10**-3

data_id = 0
data = data_dir + '/'+data_struct+'_' + str(data_id) + '.npz'
outfile_dir = '../SynData/result/'+data_struct+'/'+data_struct+'_' + str(data_id)
trained_result =  outfile_dir + '/tensorQIF_trainedW.npz'

if not os.path.exists(outfile_dir):
    os.makedirs(outfile_dir)
    
init_method = data

npzfile = np.load(data)
X_train = npzfile['X'][:train_size, :,:,:]
X_test = npzfile['X'][train_size:, :,:,:]
y_train = npzfile['y'][:train_size, :]
y_test = npzfile['y'][train_size:, :]

_, T, d1, d2 = X_train.shape
tau = T - y_train.shape[1]
M_list = Utils.get_M_list(y_train.shape[1])

W_trained, vals = minimize_tensorQIF(X_train, y_train, tau, M_list[:M_list_len], lam_list, 
                                     line_search, tol, max_step, init_method, corr_restruct, alpha,
                                     trained_record=trained_result)
np.savez(trained_result, W1=W_trained[0], W2=W_trained[1], W3=W_trained[2], W1_init=W_trained[3], W2_init=W_trained[4], W3_init=W_trained[5])

# Calculate MSE
test_size = y_test.shape[0]

X_train_flat = np.reshape(Utils.get_X_repeat(X_train,tau), [train_size*(T-tau), (tau+1)*d1*d2])
X_test_flat = np.reshape(Utils.get_X_repeat(X_test, tau), [test_size*(T-tau), (tau+1)*d1*d2])
W_vect = np.reshape(W_trained[0]+W_trained[1]+W_trained[2], [(tau+1)*d1*d2, 1])

y_pred_train = np.reshape(np.matmul(X_train_flat, W_vect), [train_size,T-tau])[:, :y_train.shape[1]]
y_pred_test = np.reshape(np.matmul(X_test_flat, W_vect), [test_size,T-tau])[:, :y_test.shape[1]]
train_MSE = np.mean(np.square(y_train - y_pred_train))
test_MSE = np.mean(np.square(y_test - y_pred_test))

print(train_MSE)
print(test_MSE)

steps = [i for i in range(len(vals))]
plt.plot(steps, vals) # show training curve
plt.show()

