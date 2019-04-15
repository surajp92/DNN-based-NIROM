#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 15:19:23 2019

@author: user1
"""
namelist = ['SEQ', 'RESNET','EULER', 'BDF2', 'BDF3']
plist = [1,2,4]
for name in namelist:
    for p in plist:
        legs = p # No. of legs = 1,2,4 
        slopenet = name # Choices: BDF2, BDF3, BDF4, SEQ, EULER, LEAPFROG, LEAPFROG-FILTER, RESNET
        problem = "ODE"
        
        import os
        if os.path.isfile('best_model.hd5'):
            os.remove('best_model.hd5')
        
        import numpy as np
        from numpy.random import seed
        seed(1)
        from tensorflow import set_random_seed
        set_random_seed(2)
        from scipy.integrate import odeint
        import matplotlib.pyplot as plt
        import pandas as pd
        from create_data_v2 import * #create_training_data_bdf2
        from model_prediction_v2 import *
        from export_data_v2 import *
        import time as cputime
        from mpl_toolkits.mplot3d import Axes3D
        
        rho = 28.0
        sigma = 10.0
        beta = 8.0 / 3.0
        
        def f(state, t):
          x, y, z = state  # unpack the state vector
          return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # derivatives
        
        state0 = [1.0, -0.5, 1.0]
        t_init  = 0.0  # Initial time
        t_final = 20.0 # Final time
        dt = 0.01
        t = np.arange(t_init, t_final, 0.01)
        
        start_ode = cputime.time()
        
        states = odeint(f, state0, t)
        
        end_ode = cputime.time()
        ode_time = end_ode-start_ode
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(states[:,0], states[:,1], states[:,2])
        plt.show()
        
        
        def coeff_determination(y_true, y_pred):
            SS_res =  K.sum(K.square( y_true-y_pred ))
            SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
            return ( 1 - SS_res/(SS_tot + K.epsilon()) )
        
        training_set = states
        m,n = training_set.shape
        n = n+1
        xtrain, ytrain = create_training_data(training_set, m, n, dt, legs, slopenet)
        
        for i in range(-4,5):
            y2s = 0.1*(i)
            state0 = [1.0, y2s, 1.0]
            states = odeint(f, state0, t)
            training_set = states
            m,n = training_set.shape
            n = n+1
            xtemp, ytemp = create_training_data(training_set, m, n, dt, legs, slopenet)
            xtrain = np.vstack((xtrain, xtemp))
            ytrain = np.vstack((ytrain, ytemp))
            
        
        start_train = cputime.time()
        
        #indices = np.random.randint(0,xtrain.shape[0],500)
        #xtrain = xtrain[indices]
        #ytrain = ytrain[indices]
        
        from sklearn.preprocessing import MinMaxScaler
        sc_input = MinMaxScaler(feature_range=(-1,1))
        sc_input = sc_input.fit(xtrain)
        xtrain_scaled = sc_input.transform(xtrain)
        xtrain_scaled.shape
        xtrain = xtrain_scaled
        
        from sklearn.preprocessing import MinMaxScaler
        sc_output = MinMaxScaler(feature_range=(-1,1))
        sc_output = sc_output.fit(ytrain)
        ytrain_scaled = sc_output.transform(ytrain)
        ytrain_scaled.shape
        ytrain = ytrain_scaled
        
        #--------------------------------------------------------------------------------------------------------------#
        from keras.models import Sequential, Model
        from keras.layers import Dense, Dropout, Input
        import keras.backend as K
        from keras.callbacks import ModelCheckpoint
        from keras.models import load_model
        from keras import optimizers
        from sklearn.preprocessing import MinMaxScaler # needed if we want to normalize input and output data. Not normalized in this code
        #--------------------------------------------------------------------------------------------------------------#
        
        # create the LSTM model
        model = Sequential()
        
        # Layers start
        input_layer = Input(shape=(legs*(n-1),))
        model.add(Dropout(0.2))
        
        # Hidden layers
        x = Dense(40, activation='relu', use_bias=True)(input_layer)
        x = Dense(40, activation='relu', use_bias=True)(x)
        #x = Dense(40, activation='relu', use_bias=True)(x)
        #x = Dense(40, activation='tanh', use_bias=True)(x)
        #x = Dense(100, activation='relu', use_bias=True)(x)
        
        op_val = Dense(3, activation='linear', use_bias=True)(x)
        custom_model = Model(inputs=input_layer, outputs=op_val)
        filepath = "best_model.hd5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        
        custom_model.compile(optimizer='adam', loss='mean_squared_error', metrics=[coeff_determination])
        history_callback = custom_model.fit(xtrain, ytrain, epochs=300, batch_size=200, verbose=1, validation_split= 0.2,
                                            callbacks=callbacks_list)
        
        # training and validation loss. Plot loss
        loss_history = history_callback.history["loss"]
        val_loss_history = history_callback.history["val_loss"]
        # evaluate the model
        scores = custom_model.evaluate(xtrain, ytrain)
        print("\n%s: %.2f%%" % (custom_model.metrics_names[1], scores[1]*100))
        
        epochs = range(1, len(loss_history) + 1)
        end_train = cputime.time()
        train_time = end_train-start_train
        
        plt.figure()
        plt.semilogy(epochs, loss_history, 'b', label='Training loss')
        plt.semilogy(epochs, val_loss_history, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()
        
        #--------------------------------------------------------------------------------------------------------------#
        # predict results recursively using the model
        state0 = [1.0, 1.0, 1.0]
        states = odeint(f, state0, t)
        testing_set = states
        m,n=testing_set.shape
        time = t
        n = n+1
        
        #--------------------------------------------------------------------------------------------------------------#
        # predict results recursively using the model 
        start_test = cputime.time()
        ytest_ml = model_predict(testing_set, m, n, dt, legs, slopenet, sigma, sc_input, sc_output)
        
           
        # sum of L2 norm of each series
        l2norm_sum, l2norm_nd = calculate_l2norm(ytest_ml, testing_set, m, n, legs, slopenet, problem)
        
        end_test = cputime.time()
        test_time = end_test - start_test
        
        list = [legs, slopenet, problem, train_time, test_time, train_time+test_time]
        with open('time.csv', 'a') as f:
            f.write("\n")
            for item in list:
                f.write("%s\t" % item)
        
        # export the solution in .csv file for further post processing
        export_results_rom(ytest_ml, testing_set, time, m, n, slopenet, legs)
        
        # plot ML prediction and true data
        plot_results_lorenz(dt,slopenet, legs)