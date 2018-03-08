# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 11:55:40 2018

@author: Ko-Shin Chen
"""

import numpy as np
import matplotlib.pyplot as plt

model = 'data'
data_struct = '500_5x5x5'
data_id = 0

trained_result = './SynData/result/'+data_struct+'/'+data_struct+'_'+str(data_id)+'/'+model+'_trainedW.npz' # training result
syn_data = './SynData/'+data_struct+'/'+data_struct+'_'+str(data_id)+'.npz' # input dataset 

cutoff = -1
# for plot color
vmax = 9
vmin = -9


single_W = True
if model == 'tensorQIF' or model == 'tf_tensorQIF':
    single_W = False

elif model == 'data':
    single_W = False
    trained_result = syn_data

npzfile = np.load(trained_result)
syn_file = np.load(syn_data)
true_W = syn_file['W1'] + syn_file['W2'] + syn_file['W3']


def unfold_0(tensor):
    '''
    Unfold a 3D tensor [d0, d1, d2] to a matrix along d0
    '''
    if tensor.shape[0] == 1:
        return tensor[0, :, :]
    
    d0, _, _ = tensor.shape
    
    return np.concatenate([tensor[i,:,:] for i in range(d0)], axis=1)

if single_W:
    trained_W = npzfile['W']
    W_unfold = unfold_0(trained_W)
    plt.imshow(W_unfold, cmap='bwr', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.title("W")
    plt.show()
    
    
else:
    W_1_unfold = unfold_0(npzfile['W1'])
    W_2_unfold = unfold_0(npzfile['W2'])
    W_3_unfold = unfold_0(npzfile['W3'])
    trained_W = npzfile['W1'] + npzfile['W2'] + npzfile['W3']
    W_unfold = unfold_0(trained_W)
    
    W_1_unfold[abs(W_1_unfold) < cutoff] = 0.0
    W_2_unfold[abs(W_2_unfold) < cutoff] = 0.0
    W_3_unfold[abs(W_3_unfold) < cutoff] = 0.0
    W_unfold[abs(W_unfold) < cutoff] = 0.0
    
    
    plt.imshow(W_1_unfold, cmap='bwr', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.title("W1")
    plt.show()
    plt.imshow(W_2_unfold, cmap='bwr', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.title("W2")
    plt.show()
    plt.imshow(W_3_unfold, cmap='bwr', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.title("W3")
    plt.show()
    plt.imshow(W_unfold, cmap='bwr', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.title("W")
    plt.show()

    
W_diff_unfold = unfold_0(abs(true_W - trained_W))
scale = np.amax(abs(true_W - trained_W))
print("W MSE = " + str(np.mean(np.square(true_W - trained_W))))
print("Max W Abs Diff = " + str(scale))
print("Min W Abs Diff = " + str(np.amin(abs(true_W - trained_W))))
plt.imshow(W_diff_unfold, cmap='bwr', interpolation='nearest', vmin=-1*scale, vmax=scale)
plt.title("abs(true_W - trained_W)")
plt.show()