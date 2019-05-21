#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:47:58 2019

@author: Suraj Pawar
"""
#namelist = ['SEQ', 'EULER']
#plist = [1,2]
#for name in namelist:
#    for p in plist:

#%%
legs = 1 # No. of legs = 1,2,4 
slopenet = 'SEQ' # Choices: BDF2, SEQ, EULER, RESNET
problem = "KO"

import os
if os.path.isfile('best_model.hd5'):
    os.remove('best_model.hd5')

#%%
import numpy as np
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
import matplotlib.pyplot as plt
from create_data import *
from model_prediction import *
from export_data import *
import time as cputime
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import optimizers
from keras import losses
from keras import regularizers
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#%%
def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


def customloss(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

#%%
def f(y, t):
    dy0dt =  y[0]   * y[2]
    dy1dt = -y[1]   * y[2]
    dy2dt = -y[0]**2 + y[1]**2
    dydt  = [dy0dt, dy1dt, dy2dt]
    return dydt

state0 = [1.0,0.05,0.0]
#state0 = 30*(np.random.rand(3)-0.5)
t_init  = 0.0  # Initial time
t_final = 10.0 # Final time
dt = 0.01
t = np.arange(t_init, t_final, dt)
nsamples = int((t_final-t_init)/dt)
states = odeint(f, state0, t)

#%%
#dataset_train = np.genfromtxt('./a94.csv', delimiter=",",skip_header=0)
training_set = states
m,n=training_set.shape

xtrain, ytrain = create_training_data(training_set, dt, legs, slopenet)

#%%
from sklearn.preprocessing import MinMaxScaler
sc_input = MinMaxScaler(feature_range=(-1,1))
sc_input = sc_input.fit(xtrain)
xtrain_scaled = sc_input.transform(xtrain)
xtrain = xtrain_scaled

from sklearn.preprocessing import MinMaxScaler
sc_output = MinMaxScaler(feature_range=(-1,1))
sc_output = sc_output.fit(ytrain)
ytrain_scaled = sc_output.transform(ytrain)
ytrain = ytrain_scaled

indices = np.random.randint(0,xtrain.shape[0],1000)
xtrainsr = xtrain[indices]
ytrainsr = ytrain[indices]


#%%
model = Sequential()

# Layers start
input_layer = Input(shape=(legs*n,))
#model.add(Dropout(0.2))

# Hidden layers
x = Dense(128, activation='relu', use_bias=True)(input_layer)
x = Dense(128, activation='relu', use_bias=True)(x)
x = Dense(128, activation='relu', use_bias=True)(x)
x = Dense(128, activation='relu', use_bias=True)(x)
#x = Dense(120, activation='relu', kernel_regularizer=regularizers.l2(0.0001),  use_bias=True)(x)
#x = Dense(120, activation='relu',  use_bias=True)(x)
#x = Dense(120, activation='relu',  use_bias=True)(x)

op_val = Dense(n, activation='linear', use_bias=True)(x)
custom_model = Model(inputs=input_layer, outputs=op_val)
filepath = "best_model.hd5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

custom_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
history_callback = custom_model.fit(xtrain, ytrain, epochs=600, batch_size=300, verbose=1, validation_split= 0.1,
                                    callbacks=callbacks_list)

# training and validation loss. Plot loss
loss_history = history_callback.history["loss"]
val_loss_history = history_callback.history["val_loss"]
# evaluate the model
scores = custom_model.evaluate(xtrain, ytrain)
print("\n%s: %.2f%%" % (custom_model.metrics_names[1], scores[1]*100))

epochs = range(1, len(loss_history) + 1)

plt.figure()
plt.semilogy(epochs, loss_history, 'b', label='Training loss')
plt.semilogy(epochs, val_loss_history, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#%% read data for testing
testing_set = states
m,n = testing_set.shape
custom_model = load_model('best_model.hd5')#,custom_objects={'coeff_determination': coeff_determination})

#%%
ytest_ml = model_predict(testing_set, dt, legs, slopenet, sc_input, sc_output)

#%%
# sum of L2 norm of each series
l2norm_sum, l2norm_nd = calculate_l2norm(ytest_ml, testing_set,legs, slopenet, problem)

# export the solution in .csv file for further post processing
export_results(ytest_ml, testing_set, t, slopenet, legs)

# plot ML prediction and true data
plot_results_ko(dt, slopenet, legs)

