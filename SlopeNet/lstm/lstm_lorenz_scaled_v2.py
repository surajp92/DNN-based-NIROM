#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 13:38:12 2019

@author: Suraj Pawar
"""
import numpy as np
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
import pandas as pd
from scipy.integrate import odeint

from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import LSTM

from scipy.integrate import odeint
import matplotlib.pyplot as plt
from random import uniform
from create_data_v2 import * 
from model_prediction_v2 import *
from export_data_v2 import *

# Lorenz system
rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0

def f(state, t):
  x, y, z = state  # unpack the state vector
  return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # derivatives

state0 = [1.508870,-1.531271, 25.46091]
t_init  = 0.0  # Initial time
t_final = 20.0 # Final time
dt = 0.005
t = np.arange(t_init, t_final, dt)
states = odeint(f, state0, t)
nsamples = int((t_final-t_init)/dt)

training_set = states[0:int(nsamples/2.0),:]
m,n = training_set.shape
lookback = 20
problem = "ROM"

slopenet = "LSTM"
legs = lookback

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)
training_set_scaled.shape
training_set = training_set_scaled

xtrain, ytrain = create_training_data_lstm(training_set, m, n, lookback)

#from sklearn.preprocessing import MinMaxScaler
#sc_input = MinMaxScaler(feature_range=(-1,1))
#sc_input = sc_input.fit(xtrain)
#xtrain_scaled = sc_input.transform(xtrain)
#xtrain_scaled.shape
#xtrain = xtrain_scaled
#
#from sklearn.preprocessing import MinMaxScaler
#sc_output = MinMaxScaler(feature_range=(-1,1))
#sc_output = sc_output.fit(ytrain)
#ytrain_scaled = sc_output.transform(ytrain)
#ytrain_scaled.shape
#ytrain = ytrain_scaled

# create the LSTM model
model = Sequential()
#model.add(LSTM(3, input_shape=(1, 3), return_sequences=True, activation='tanh'))
#model.add(LSTM(120, input_shape=(lookback, n), return_sequences=True, activation='tanh'))
model.add(LSTM(40, input_shape=(lookback, n), return_sequences=True, activation='tanh'))
model.add(LSTM(40, input_shape=(lookback, n), return_sequences=True, activation='tanh'))
model.add(LSTM(40, input_shape=(lookback, n), activation='tanh'))
model.add(Dense(n))

# compile the model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# run the model
history = model.fit(xtrain, ytrain, nb_epoch=400, batch_size=50, validation_split=0.3)

# training and validation loss. Plot loss
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.figure()
plt.semilogy(epochs, loss, 'b', label='Training loss')
plt.semilogy(epochs, val_loss, 'r', label='Validationloss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#--------------------------------------------------------------------------------------------------------------#
#read data for testing
testing_set = states

testing_set_scaled = sc.fit_transform(testing_set)
testing_set_scaled.shape
testing_set= testing_set_scaled


m,n = testing_set.shape
ytest = np.zeros((1,lookback,n))
ytest_ml = np.zeros((m,n))
# create input at t= 0 for the model testing
for i in range(lookback):
    ytest[0,i,:] = testing_set[i]
    ytest_ml[i] = testing_set[i]

for i in range(lookback,m):
    ytest_sc = ytest
    slope_ml = model.predict(ytest_sc) # slope from LSTM/ ML model
    slope_ml_sc = slope_ml
    ytest_ml[i] = slope_ml_sc
    e = ytest
    for j in range(lookback-1):
        e[0,j,:] = e[0,j+1,:]
    e[0,lookback-1,:] = slope_ml
    ytest = e 

# predict results recursively using the model 
n = n+1 # this is how export data code is written

ytest_ml_unscaled = sc.inverse_transform(ytest_ml)
ytest_ml_unscaled.shape
ytest_ml= ytest_ml_unscaled

testing_set_unscaled = sc.inverse_transform(testing_set)
testing_set_unscaled.shape
testing_set= testing_set_unscaled

# sum of L2 norm of each series
#l2norm_sum, l2norm_nd = calculate_l2norm(ytest_ml, testing_set, m, n, lookback, "LSTM", problem)

# export the solution in .csv file for further post processing
export_results_rom(ytest_ml, testing_set, t, m, n, slopenet, legs)

# plot ML prediction and true data
plot_results_lorenz(dt, slopenet, legs)



