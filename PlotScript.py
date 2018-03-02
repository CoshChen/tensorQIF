# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 11:55:40 2018

@author: Ko-Shin Chen
"""

import numpy as np
import matplotlib.pyplot as plt

data_id = 5
data = './SynData/result/500_5x5x5/500_5x5x5_'+str(data_id)+'/tf_tensorQIF_trainedW.npz' # training result
#data = './SynData/500_5x5x5/500_5x5x5_'+str(data_id)+'.npz' # input dataset 
#data = './TestCode/trainedW_400.npz' # test dataset

npzfile = np.load(data)
# variables =  npzfile.files # Check variable names in file

cutoff = -1
# for plot color
vmax = 5
vmin = -5


def unfold_0(tensor):
    '''
    Unfold a 3D tensor [d0, d1, d2] to a matrix along d0
    '''
    if tensor.shape[0] == 1:
        return tensor[0, :, :]
    
    d0, _, _ = tensor.shape
    
    return np.concatenate([tensor[i,:,:] for i in range(d0)], axis=1)


W_1_unfold = unfold_0(npzfile['W1'])
W_2_unfold = unfold_0(npzfile['W2'])
W_3_unfold = unfold_0(npzfile['W3'])
W_unfold = unfold_0(npzfile['W1']+npzfile['W2']+npzfile['W3'])

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

