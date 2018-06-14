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


epoch = 2* 10 ** 4
report = 10 ** 3
tol = 10 ** -12
tol_2 = 10 ** -12
train_size = 300

init_lam = 0.1
c = np.sqrt(3/2)

data_struct = '1000_3x2x2_AR08_c4'
dataset_num = 100
data_dir = '../SynData/' + str(data_struct)
M_list_ind = [0, 1, 2]

for data_id in range(dataset_num):
    print("--- Data # " + str(data_id) + " ---")
    data = data_dir + '/' + data_struct + '_' + str(data_id) + '.npz'
    outfile_dir = '../SynData/result/'+data_struct +'/m'+str(train_size)+'/'+data_struct +'_'+str(data_id)
    W_outfile = outfile_dir + '/tf_tensorQIF_trainedW.npz'
    initial_state = outfile_dir + '/tf_tensorQIF_initW.npz'

    if not os.path.exists(outfile_dir):
        os.makedirs(outfile_dir)

    '''
    Load Data and Set Up Dimensions
    '''
    npzfile = np.load(data)  # Load Data
    X = npzfile['X'][:train_size, :, :, :]
    y = npzfile['y'][:train_size, :]

    _, T, d1, d2 = X.shape
    tau = T - y.shape[1]
    M_list = Utils.get_M_list(y.shape[1])  # T-tau
    
    cv_train_size = int(train_size*0.8)

    X_repeat_train = Utils.get_X_repeat(X, tau)[:cv_train_size, :, :]
    labels_train = y[:cv_train_size, :]
    X_repeat_vali = Utils.get_X_repeat(X, tau)[cv_train_size:, :, :]
    labels_vali = y[cv_train_size:, :]
    
    '''
    Setup TensorFlow Graph
    '''
    tf.reset_default_graph()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)

    '''
    Model Graph
    '''
    X_repeat = tf.placeholder(dtype='float64', shape=[None, T - tau, (tau + 1) * d1 * d2], name='X_repeat')
    labels = tf.placeholder(dtype='float64', shape=[None, T - tau], name='labels')
    lam = tf.placeholder(dtype='float64', shape=(), name='lambda')

    M = []
    for i in M_list_ind:
        M.append(tf.constant(M_list[i]))

    W_list_1 = []
    W_list_2 = []
    W_list_3 = []

    print("Random Initial Values")
    for t in range(tau + 1):
        W_list_1.append(
            tf.Variable(tf.truncated_normal([d1, d2], mean=0.0, stddev=0.01, dtype='float64'), name='W1_' + str(t)))
    for j in range(d1):
        W_list_2.append(tf.Variable(tf.truncated_normal([tau + 1, d2], mean=0.0, stddev=0.01, dtype='float64'),
                                    name='W2_' + str(j)))
    for j in range(d2):
        W_list_3.append(tf.Variable(tf.truncated_normal([tau + 1, d1], mean=0.0, stddev=0.01, dtype='float64'),
                                    name='W3_' + str(j)))

    W1 = tf.stack(W_list_1, axis=0, name='W1')
    W2 = tf.stack(W_list_2, axis=1, name='W2')
    W3 = tf.stack(W_list_3, axis=2, name='W3')
    W = tf.add(tf.add(W1, W2), W3, name='W')

    reg = tf.add(
        tf.add(qif.tensor_L12(lam, W_list_1, 'W_1_norm'), qif.tensor_L12(c*lam, W_list_2, 'W_2_norm')),
        qif.tensor_L12(c*lam, W_list_3, 'W_3_norm'), name='reg')
    QIF = qif.QIF_Gaussian_tf(X_repeat, labels, M, W)
    loss = tf.add(QIF, reg, name='loss')

    s = tf.get_default_graph().get_tensor_by_name('s:0')  # y-mu [batch, T-tau]
    MSE = tf.reduce_mean(tf.square(s), name='MSE')


    '''
    Optimizer
    '''
    train = tf.train.AdamOptimizer().minimize(loss)
    init = tf.global_variables_initializer()
    sess.run(init)
    
    np.savez(initial_state, W1_init=W1.eval(session=sess), W2_init=W2.eval(session=sess), W3_init=W3.eval(session=sess))


    '''
    Feed Dictionary
    '''
    feed_train = {X_repeat: X_repeat_train, labels: labels_train, lam: init_lam}
    feed_vali = {X_repeat: X_repeat_vali, labels: labels_vali, lam: init_lam}
    
    
    pre_loss = loss.eval(session=sess, feed_dict=feed_train)
    pre_MSE = MSE.eval(session=sess, feed_dict=feed_vali)
    diff = 1.0
    diff_2 = 1.0

    print("Initial loss = " + str(pre_loss))
    print(" ")

    for i in range(epoch):
        sess.run(train, feed_dict=feed_train)

        if (i + 1) % report == 0:
            curr_loss = loss.eval(session=sess, feed_dict=feed_train)
            curr_MSE = MSE.eval(session=sess, feed_dict=feed_vali)
            diff = abs(pre_loss - curr_loss)
            diff_2 = abs(pre_MSE - curr_MSE)

            np.savez(W_outfile, W1=W1.eval(session=sess), W2=W2.eval(session=sess), W3=W3.eval(session=sess), lam = init_lam)

            print("Step " + str(i) + ":" + str(curr_loss))
            print("Training MSE: " + str(MSE.eval(session=sess, feed_dict=feed_train)))
            print("Vali MSE: " + str(MSE.eval(session=sess, feed_dict=feed_vali)))
            print(" ")
            
            pre_loss = curr_loss
            pre_MSE = curr_MSE

            if diff < tol or diff_2 < tol_2:
                print("Small Function Value Change: " + str(diff < tol))
                print("Small Training MSE Change: " + str(diff_2 < tol_2))
                print("Early Stop")
                break
    print(" ")

    W1_zero = W1.eval(session=sess)
    W2_zero = W2.eval(session=sess)
    W3_zero = W3.eval(session=sess)
    final_MSE = pre_MSE
    final_lam = init_lam
    
    print("Initial CV MSE = " + str(final_MSE))
    print(" ")
    
    for lam_temp in [1] + [l for l in range(10, 100, 10)] + [l for l in range(100, 200, 20)] \
            + [l for l in range(200, 400, 50)]:
        feed_train = {X_repeat: X_repeat_train, labels: labels_train, lam: lam_temp}
        feed_vali = {X_repeat: X_repeat_vali, labels: labels_vali, lam: lam_temp}

        sess.run(init)

        for t in range(tau + 1):
            sess.run(tf.assign(tf.get_default_graph().get_tensor_by_name('W1_'+str(t)+':0'), W1_zero[t,:,:]))
        for j in range(d1):
            sess.run(tf.assign(tf.get_default_graph().get_tensor_by_name('W2_'+str(j)+':0'), W2_zero[:, j, :]))
        for j in range(d2):
            sess.run(tf.assign(tf.get_default_graph().get_tensor_by_name('W3_'+str(j)+':0'), W3_zero[:, :, j]))
        
        pre_loss = loss.eval(session=sess, feed_dict=feed_train)
        pre_MSE = MSE.eval(session=sess, feed_dict=feed_vali)
        
        print("Lambda = " + str(lam_temp))
        print("Inner Loop Initial Loss = " + str(pre_loss))
        print("Inner Loop Initial MSE = " + str(pre_MSE))
        print(" ")
        
        for i in range(epoch):
            sess.run(train, feed_dict=feed_train)
    
            if (i+1) % report == 0:
                curr_loss = loss.eval(session=sess, feed_dict=feed_train)
                curr_MSE = MSE.eval(session=sess, feed_dict=feed_vali)
                diff = abs(pre_loss - curr_loss)
                diff_2 = abs(pre_MSE - curr_MSE)
                
                print("Step " + str(i) + ":" + str(curr_loss))
                print("Training MSE: " + str(MSE.eval(session=sess, feed_dict=feed_train)))
                print("Vali MSE: " + str(curr_MSE))
                print(" ")
                
                pre_loss = curr_loss
                pre_MSE = curr_MSE
    
                if diff < tol or diff_2 < tol_2:
                    print("Small Function Value Change: " + str(diff < tol))
                    print("Small Training MSE Change: " + str(diff_2 < tol_2))
                    print("Early Stop")
                    print(" ")
                    break           
            
        if pre_MSE > final_MSE:
            print("Final MSE = " + str(final_MSE))
            print("Final Lambda = " + str(final_lam))
            print(" ")
            break
            
        print("Update Ws")
        np.savez(W_outfile, W1=W1.eval(session=sess), W2=W2.eval(session=sess), W3=W3.eval(session=sess), lam = lam_temp)
        final_MSE = pre_MSE
        final_lam = lam_temp
        
        print(" ")
        
        