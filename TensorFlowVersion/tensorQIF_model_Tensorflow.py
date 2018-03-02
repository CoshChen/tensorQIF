# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 12:58:39 2018

@author: Ko-Shin Chen
"""

import numpy as np
import tensorflow as tf
import os
import Utils
import QIF_Tensorflow as qif

tf.reset_default_graph()
config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)

data_id = 0
data = '../SynData/500_5x5x5/500_5x5x5_' + str(data_id) + '.npz'
outfile_dir = '../SynData/result/500_5x5x5/500_5x5x5_' + str(data_id)
check_point_file = outfile_dir + '/tf_tensorQIF.ckpt'
check_point_file_to_load = check_point_file
W_outfile = outfile_dir + '/tf_tensorQIF_trainedW.npz'
initial_state = outfile_dir + '/tf_tensorQIF_initW.npz'

epoch = 10**7
report = 10**3
tol = 10**-12
train_size = 400

lam_list = [0.3, 0.3, 0.3]

if not os.path.exists(outfile_dir):
    os.makedirs(outfile_dir)


'''
Load Data and Set Up Dimensions
'''
npzfile = np.load(data)
X = npzfile['X']
y = npzfile['y']

batch, T, d1, d2 = X.shape
tau = T - y.shape[1]
M_list = Utils.get_M_list(y.shape[1]) # T-tau

X_repeat_train = Utils.get_X_repeat(X, tau)[:train_size, : ,:]
labels_train = y[:train_size, :]
X_repeat_test = Utils.get_X_repeat(X, tau)[train_size:, : ,:]
labels_test = y[train_size:, :]


'''
Model Graph
'''
if not os.path.exists(check_point_file_to_load + '.meta'):
    print("Building new model graph.")
    
    new_model = True
    
    X_repeat = tf.placeholder(dtype='float64', shape = [None, T-tau, (tau+1)*d1*d2], name='X_repeat')
    labels = tf.placeholder(dtype='float64', shape = [None, T-tau], name='labels')
    
    M = []
    M.append(tf.constant(M_list[0]))
    M.append(tf.constant(M_list[1]))
    
    W_list_1 = []
    W_list_2 = []
    W_list_3 = []
    
    for t in range(tau+1):
        W_list_1.append(tf.Variable(tf.truncated_normal([d1, d2], mean = 0.0, stddev = 0.01, dtype='float64'), name='W1_'+str(t)))
        # W_list_1.append(tf.Variable(tf.zeros([d1, d2], dtype='float64'), name='W1_'+str(t)))
    for j in range(d1):
        W_list_2.append(tf.Variable(tf.truncated_normal([tau+1, d2], mean = 0.0, stddev = 0.01, dtype='float64'), name='W2_'+str(j)))
        # W_list_2.append(tf.Variable(tf.zeros([tau+1, d2], dtype='float64'), name='W2_'+str(j)))
    for j in range(d2):
        W_list_3.append(tf.Variable(tf.truncated_normal([tau+1, d1], mean = 0.0, stddev = 0.01, dtype='float64'), name='W3_'+str(j)))
        # W_list_3.append(tf.Variable(tf.zeros([tau+1, d1], dtype='float64'), name='W3_'+str(j)))
    
    W1 = tf.stack(W_list_1, axis=0, name='W1')
    W2 = tf.stack(W_list_2, axis=1, name='W2')
    W3 = tf.stack(W_list_3, axis=2, name='W3')
    W = tf.add(tf.add(W1, W2), W3, name='W')
    
    reg = tf.add(tf.add(qif.tensor_L12(lam_list[0], W_list_1, 'W_1_norm'), qif.tensor_L12(lam_list[1], W_list_2, 'W_2_norm')), qif.tensor_L12(lam_list[2], W_list_3, 'W_3_norm'), name='reg')
    QIF = qif.QIF_Gaussian_tf(X_repeat, labels, tau, M, W)
    loss = tf.add(QIF, reg, name='loss')
    
    s = tf.get_default_graph().get_tensor_by_name('s:0') # y-mu [batch, T-tau]
    MSE = tf.reduce_mean(tf.square(s), name='MSE')
    
    saver = tf.train.Saver()
    
else:
    print("Reloading existing model")
    new_model = False
    
    saver = tf.train.import_meta_graph(check_point_file_to_load + '.meta')
    
    X_repeat = tf.get_default_graph().get_tensor_by_name('X_repeat:0')
    labels = tf.get_default_graph().get_tensor_by_name('labels:0')
    
    W1 = tf.get_default_graph().get_tensor_by_name('W1:0')
    W2 = tf.get_default_graph().get_tensor_by_name('W2:0')
    W3 = tf.get_default_graph().get_tensor_by_name('W3:0')
    
    reg = tf.get_default_graph().get_tensor_by_name('reg:0')
    QIF = tf.get_default_graph().get_tensor_by_name('QIF:0')
    loss = tf.get_default_graph().get_tensor_by_name('loss:0')
    
    MSE = tf.get_default_graph().get_tensor_by_name('MSE:0')
    

'''
Optimizer
'''
if new_model:
    train = tf.train.AdamOptimizer().minimize(loss)
    init = tf.global_variables_initializer()
    sess.run(init)
    tf.add_to_collection('train', train)
    
else:
    train = tf.get_collection('train')[0]
    init = tf.global_variables_initializer()
    sess.run(init)
    saver.restore(sess, check_point_file_to_load)
    

feed_train = {X_repeat: X_repeat_train, labels: labels_train}
feed_test = {X_repeat: X_repeat_test, labels: labels_test}


if new_model:
    # Save initial Wi
    np.savez(initial_state, W1_init=W1.eval(session=sess), W2_init=W2.eval(session=sess), W3_init=W3.eval(session=sess))

pre = loss.eval(session=sess, feed_dict=feed_train)
diff = 1.0
step = 0

print("Initial loss = " + str(pre))
print(" ")  

for i in range(epoch):
    sess.run(train, feed_dict=feed_train)
    curr = loss.eval(session=sess, feed_dict=feed_train)
    
    if (i+1)%report == 0:
        saver.save(sess, check_point_file)
        np.savez(W_outfile, W1=W1.eval(session=sess), W2=W2.eval(session=sess), W3=W3.eval(session=sess))
        print("Step " + str(i) + ":" + str(curr))
        print("Training MSE: " + str(MSE.eval(session=sess, feed_dict=feed_train)))
        print("Test MSE: " + str(MSE.eval(session=sess, feed_dict=feed_test)))
        print(" ")
        
    diff = abs(pre - curr)
    
    if diff < tol:
        saver.save(sess, check_point_file)
        np.savez(W_outfile, W1=W1.eval(session=sess), W2=W2.eval(session=sess), W3=W3.eval(session=sess))        
        print("Early Stop")
        print(diff)
        print("Step " + str(i) + ":" + str(curr))
        print("Training MSE: " + str(MSE.eval(session=sess, feed_dict=feed_train)))
        print("Test MSE: " + str(MSE.eval(session=sess, feed_dict=feed_test)))
        break
    
    pre = curr

print("Done epoch = " + str(epoch))  

if epoch%report != 0:
    saver.save(sess, check_point_file)
    np.savez(W_outfile, W1=W1.eval(session=sess), W2=W2.eval(session=sess), W3=W3.eval(session=sess))
    print("Step " + str(i) + ":" + str(pre))
    print("Training MSE: " + str(MSE.eval(session=sess, feed_dict=feed_train)))
    print("Test MSE: " + str(MSE.eval(session=sess, feed_dict=feed_test)))

