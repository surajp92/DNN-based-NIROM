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
set_random_seed(2)

import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler # needed if we want to normalize input and output data. Not normalized in this code

from scipy.integrate import odeint
import matplotlib.pyplot as plt
from random import uniform
from create_data_v2 import * 
from model_prediction_v2 import *
from export_data_v2 import *

# function that returns dy/dt
def odemodel(y,t):
    dy0dt =  y[0]   * y[2]
    dy1dt = -y[1]   * y[2]
    dy2dt = -y[0]**2 + y[1]**2
    dydt  = [dy0dt, dy1dt, dy2dt]
    return dydt

# time points
t_init  = 0.0  # Initial time
t_final = 10.0 # Final time
nt_steps = 200 # Number of time steps
t = np.linspace(0,t_final, num=nt_steps)
dt = (t_final - t_init)/nt_steps
lookback = 4
problem = "ODE"
# initial condition
y0 = [1, -0.1, 0]
# solve ODE
training_set = odeint(odemodel,y0,t)
# Note more sophisticated ode integrators should be used
m,n = training_set.shape
xtrain, ytrain = create_training_data_lstm(training_set, m, n, lookback)

# additional data for training with random initial condition
for i in range(-9,11):
    y2s = 0.1*(0.1*i)
    y0 = [1.0, y2s, 0.0]
    training_set = odeint(odemodel,y0,t)
    m,n = training_set.shape
    xtemp, ytemp= create_training_data_lstm(training_set, m, n, lookback)
    xtrain = np.vstack((xtrain, xtemp))
    ytrain = np.vstack((ytrain, ytemp))


# create the LSTM model
model = Sequential()
#model.add(LSTM(3, input_shape=(1, 3), return_sequences=True, activation='tanh'))
#model.add(LSTM(12, input_shape=(1, 3), return_sequences=True, activation='tanh'))
#model.add(LSTM(12, input_shape=(1, 3), return_sequences=True, activation='tanh'))
model.add(LSTM(60, input_shape=(lookback, 3), activation='tanh'))
model.add(Dense(3))

# compile the model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# run the model
history = model.fit(xtrain, ytrain, nb_epoch=500, batch_size=100, validation_split=0.3)

# training and validation loss. Plot loss
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validationloss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

y2s = 0.1*0.25
y0test = [1, y2s, 0]
# solve ODE
testing_set1 = odeint(odemodel,y0test,t)
m,n = testing_set1.shape
n = n+1
ytest = np.zeros((1,lookback,3))
ytest_ml1 = np.zeros((nt_steps,3))
# create input at t= 0 for the model testing
for i in range(lookback):
    ytest[0,i,:] = testing_set1[i]
    ytest_ml1[i] = testing_set1[i]

for i in range(lookback,nt_steps):
    slope_ml = model.predict(ytest) # slope from LSTM/ ML model
    ytest_ml1[i] = slope_ml
    e = ytest
    for i in range(lookback-1):
        e[0,i,:] = e[0,i+1,:]
    e[0,lookback-1,:] = slope_ml
    ytest = e 

# predict results recursively using the model 
#ytest_ml = model_predict(testing_set, m, n, dt, legs, slopenet)

# sum of L2 norm of each series
l2norm_sum1, l2norm_nd1 = calculate_l2norm(ytest_ml1, testing_set1, m, n, lookback, "LSTM", problem, y2s)


y2s = -0.1*0.25
y0test = [1, y2s, 0]
# solve ODE
testing_set2 = odeint(odemodel,y0test,t)
m,n = testing_set2.shape
n = n+1
ytest = np.zeros((1,lookback,3))
ytest_ml2 = np.zeros((nt_steps,3))
# create input at t= 0 for the model testing
for i in range(lookback):
    ytest[0,i,:] = testing_set2[i]
    ytest_ml2[i] = testing_set2[i]

for i in range(lookback,nt_steps):
    slope_ml = model.predict(ytest) # slope from LSTM/ ML model
    ytest_ml2[i] = slope_ml
    e = ytest
    for i in range(lookback-1):
        e[0,i,:] = e[0,i+1,:]
    e[0,lookback-1,:] = slope_ml
    ytest = e 

# predict results recursively using the model 
#ytest_ml = model_predict(testing_set, m, n, dt, legs, slopenet)

# sum of L2 norm of each series
l2norm_sum2, l2norm_nd2 = calculate_l2norm(ytest_ml2, testing_set2, m, n, lookback, "LSTM", problem, y2s)

# export the solution in .csv file for further post processing
ytest_ml = np.hstack((ytest_ml1, ytest_ml2))
testing_set = np.hstack((testing_set1, testing_set2))
export_results_ode(ytest_ml, testing_set, m, n)

# plot ML prediction and true data
plot_results_ode()



