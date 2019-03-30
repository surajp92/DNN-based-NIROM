#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:47:58 2019

@author: Suraj Pawar
"""
import numpy as np
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import matplotlib.pyplot as plt
import pandas as pd
from create_data_v2 import * #create_training_data_bdf2
from model_prediction_v2 import *
from export_data_v2 import *

#--------------------------------------------------------------------------------------------------------------#
def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

#--------------------------------------------------------------------------------------------------------------#
dataset_train = pd.read_csv('./c.csv', sep=",",skiprows=0,header = None, nrows=1000)
m,n=dataset_train.shape
training_set = dataset_train.iloc[:,0:n].values
time = training_set[:,0]
dt = training_set[1,0] - training_set[0,0]
training_set = training_set[:,1:n]

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)
training_set_scaled.shape
training_set = training_set_scaled

legs = 4 # No. of legs = 1,2,4 
slopenet = "BDF4" # Choices: BDF2, BDF3, BDF4, SEQ, EULER, LEAPFROG, LEAPFROG-FILTER
sigma = 0.1
problem = "ROM"

xtrain, ytrain = create_training_data(training_set, m, n, dt, legs, slopenet)

#--------------------------------------------------------------------------------------------------------------#
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import optimizers

#--------------------------------------------------------------------------------------------------------------#
# create the LSTM model
model = Sequential()

# Layers start
input_layer = Input(shape=(legs*(n-1),))
#model.add(Dropout(0.2))

# Hidden layers
x = Dense(100, activation='tanh', use_bias=True)(input_layer)
x = Dense(100, activation='tanh', use_bias=True)(x)
#x = Dense(100, activation='tanh', use_bias=True)(x)
#x = Dense(100, activation='relu', use_bias=True)(x)
#x = Dense(100, activation='relu', use_bias=True)(x)

op_val = Dense(n-1, activation='linear', use_bias=True)(x)
custom_model = Model(inputs=input_layer, outputs=op_val)
filepath = "best_model.hd5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

custom_model.compile(optimizer='adam', loss='mean_squared_error', metrics=[coeff_determination])
history_callback = custom_model.fit(xtrain, ytrain, epochs=900, batch_size=100, verbose=1, validation_split= 0.1,
                                    callbacks=callbacks_list)

#--------------------------------------------------------------------------------------------------------------#
# training and validation loss. Plot loss
loss_history = history_callback.history["loss"]
val_loss_history = history_callback.history["val_loss"]
# evaluate the model
scores = custom_model.evaluate(xtrain, ytrain)
print("\n%s: %.2f%%" % (custom_model.metrics_names[1], scores[1]*100))

epochs = range(1, len(loss_history) + 1)

plt.figure()
plt.plot(epochs, loss_history, 'b', label='Training loss')
plt.plot(epochs, val_loss_history, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#--------------------------------------------------------------------------------------------------------------#
#read data for testing
dataset_test = pd.read_csv('./c.csv', sep=",",header = None, skiprows=0, nrows=2000)
dataset_total = pd.concat((dataset_train,dataset_test),axis=0)
dataset_total.drop(dataset_total.columns[[0]], axis=1, inplace=True)
m,n=dataset_test.shape
testing_set = dataset_test.iloc[:,0:n].values
time = testing_set[:,0]
testing_set = testing_set[:,1:n]

testing_set_scaled = sc.fit_transform(testing_set)
testing_set_scaled.shape
testing_set= testing_set_scaled

#--------------------------------------------------------------------------------------------------------------#
# predict results recursively using the model 
ytest_ml = model_predict(testing_set, m, n, dt, legs, slopenet, sigma)

ytest_ml_unscaled = sc.inverse_transform(ytest_ml)
ytest_ml_unscaled.shape
ytest_ml= ytest_ml_unscaled

testing_set_unscaled = sc.inverse_transform(testing_set)
testing_set_unscaled.shape
testing_set= testing_set_unscaled


# sum of L2 norm of each series
l2norm_sum, l2norm_nd = calculate_l2norm(ytest_ml, testing_set, m, n, legs, slopenet, problem)

# export the solution in .csv file for further post processing
export_results_rom(ytest_ml, testing_set, time, m, n)

# plot ML prediction and true data
plot_results_rom()

