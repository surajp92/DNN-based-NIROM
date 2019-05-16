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
import pandas as pd

import math
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


dataset_train = pd.read_csv('./a.csv', sep=",",skiprows=0,header = None, nrows=1000)
m,n=dataset_train.shape
training_set = dataset_train.iloc[:,0:n].values
dt = training_set[1,0] - training_set[0,0]
training_set = training_set[:,1:n]
n = n-1
lookback = 8
problem = "ROM"

slopenet = "LSTM"
legs = lookback

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)
training_set_scaled.shape
training_set = training_set_scaled

xtrain, ytrain = create_training_data_lstm(training_set, m, n, lookback)

# create the LSTM model
model = Sequential()
#model.add(LSTM(3, input_shape=(1, 3), return_sequences=True, activation='tanh'))
#model.add(LSTM(12, input_shape=(1, 3), return_sequences=True, activation='tanh'))
#model.add(LSTM(60, input_shape=(lookback, n), return_sequences=True, activation='tanh'))
model.add(LSTM(120, input_shape=(lookback, n), activation='tanh'))
model.add(Dense(n))

# compile the model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# run the model
history = model.fit(xtrain, ytrain, nb_epoch=500, batch_size=50, validation_split=0.3)

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
dataset_test = pd.read_csv('./a.csv', sep=",",header = None, skiprows=0)
dataset_total = pd.concat((dataset_train,dataset_test),axis=0)
dataset_total.drop(dataset_total.columns[[0]], axis=1, inplace=True)
m,n=dataset_test.shape
testing_set = dataset_test.iloc[:,0:n].values
time = testing_set[:,0]
testing_set = testing_set[:,1:n]

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
    slope_ml = model.predict(ytest) # slope from LSTM/ ML model
    ytest_ml[i] = slope_ml
    e = ytest
    for i in range(lookback-1):
        e[0,i,:] = e[0,i+1,:]
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
l2norm_sum, l2norm_nd = calculate_l2norm(ytest_ml, testing_set, m, n, lookback, "LSTM", problem)

# export the solution in .csv file for further post processing
export_results_rom(ytest_ml, testing_set, time, m, n, slopenet, legs)

# plot ML prediction and true data
plot_results_rom(dt, slopenet, legs)



