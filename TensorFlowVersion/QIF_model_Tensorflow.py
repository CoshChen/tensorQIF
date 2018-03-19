# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 14:45:24 2018

@author: Ko-Shin Chen
"""

import numpy as np
import tensorflow as tf
import os
import Utils
import QIF_Tensorflow as qif

retrain = True

epoch = 10 ** 5
report = 10 ** 3
tol = 10 ** -12
tol_2 = 10 ** -6
train_size = 400


data_struct = '500_5x5x5'
dataset_num = 100
data_dir = '../SynData/' + str(data_struct)
M_list_ind = [0, 1]

for data_id in range(dataset_num):
    print("--- Data # " + str(data_id) + " ---")
    data = data_dir + '/' + data_struct + '_' + str(data_id) + '.npz'
    outfile_dir = '../SynData/result/' + data_struct + '/' + data_struct + '_' + str(data_id)
    check_point_file = outfile_dir + '/tf_QIF.ckpt'
    check_point_file_to_load = check_point_file
    W_outfile = outfile_dir + '/tf_QIF_trainedW.npz'
    initial_state = outfile_dir + '/tf_QIF_initW.npz'

    if not os.path.exists(outfile_dir):
        os.makedirs(outfile_dir)

    '''
    Initial Method
    '''
    init_method = 'zeros' # Options: 'zeros' or 'path to initial states from other training result'
    

    '''
    Load Data and Set Up Dimensions
    '''
    npzfile = np.load(data)  # Load Data
    X = npzfile['X']
    y = npzfile['y']

    batch, T, d1, d2 = X.shape
    tau = T - y.shape[1]
    M_list = Utils.get_M_list(y.shape[1])  # T-tau

    X_train = np.reshape(X[:train_size, :(T-tau), :, :], [train_size, T-tau, d1*d2]) # only use time points t < T-tau
    labels_train = y[:train_size, :]
    X_test = np.reshape(X[train_size:, :(T-tau), :, :], [batch - train_size, T-tau, d1*d2])
    labels_test = y[train_size:, :]

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
    if retrain or not os.path.exists(check_point_file_to_load + '.meta'):
        print("Building new model graph.")

        new_model = True

        X = tf.placeholder(dtype='float64', shape=[None, T-tau, d1*d2], name='X')
        labels = tf.placeholder(dtype='float64', shape=[None, T-tau], name='labels')

        M = []
        for i in M_list_ind:
            M.append(tf.constant(M_list[i]))

        if init_method and os.path.exists(init_method):
            print("Use initial states from other training result.")
            W = tf.Variable(np.load(init_method)['W_init'], dtype='float64', name='W')
        
        elif init_method == 'zeros':
            print("Use Zeros as Initital Values")
            W = tf.Variable(tf.zeros([1, d1, d2], dtype='float64'), name='W')

        else:
            print("Random Initial Values")
            W = tf.Variable(tf.truncated_normal([1, d1, d2], mean=0.0, stddev=0.01, dtype='float64'), name='W')
            
        QIF = qif.QIF_Gaussian_tf(X, labels, M, W)
        s = tf.get_default_graph().get_tensor_by_name('s:0')  # y-mu [batch, T-tau]
        MSE = tf.reduce_mean(tf.square(s), name='MSE')

        saver = tf.train.Saver()

    else:
        print("Reloading existing model")
        new_model = False

        saver = tf.train.import_meta_graph(check_point_file_to_load + '.meta')

        X = tf.get_default_graph().get_tensor_by_name('X:0')
        labels = tf.get_default_graph().get_tensor_by_name('labels:0')

        W = tf.get_default_graph().get_tensor_by_name('W:0')
        QIF = tf.get_default_graph().get_tensor_by_name('QIF:0')
        MSE = tf.get_default_graph().get_tensor_by_name('MSE:0')

    '''
    Optimizer
    '''
    if new_model:
        train = tf.train.AdamOptimizer().minimize(QIF)
        init = tf.global_variables_initializer()
        sess.run(init)
        tf.add_to_collection('train', train)

    else:
        train = tf.get_collection('train')[0]
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, check_point_file_to_load)

    '''
    Feed Dictionary
    '''
    feed_train = {X: X_train, labels: labels_train}
    feed_test = {X: X_test, labels: labels_test}

    if new_model and not init_method:
        # Save initial W
        np.savez(initial_state, W_init=W.eval(session=sess))

    pre_loss = QIF.eval(session=sess, feed_dict=feed_train)
    pre_MSE = MSE.eval(session=sess, feed_dict=feed_test)
    diff = 1.0
    diff_2 = 1.0

    print("Initial loss = " + str(pre_loss))
    print(" ")

    for i in range(epoch):
        sess.run(train, feed_dict=feed_train)

        if (i + 1) % report == 0:
            curr_loss = QIF.eval(session=sess, feed_dict=feed_train)
            curr_MSE = MSE.eval(session=sess, feed_dict=feed_test)
            diff = abs(pre_loss - curr_loss)
            diff_2 = abs(pre_MSE - curr_MSE)

            saver.save(sess, check_point_file)
            np.savez(W_outfile, W=W.eval(session=sess))

            print("Step " + str(i) + ":" + str(curr_loss))
            print("Training MSE: " + str(MSE.eval(session=sess, feed_dict=feed_train)))
            print("Test MSE: " + str(MSE.eval(session=sess, feed_dict=feed_test)))
            print(" ")

            if diff < tol or diff_2 < tol_2:
                print("Small Function Value Change: " + str(diff < tol))
                print("Small MSE Change: " + str(diff_2 < tol_2))
                print("Early Stop")
                break

            pre_loss = curr_loss
            pre_MSE = curr_MSE

    print(" ")

