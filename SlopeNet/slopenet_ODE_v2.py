#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 15:19:23 2019

@author: user1
"""
import numpy as np
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
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

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


# time points
t_init  = 0.0  # Initial time
t_final = 10.0 # Final time
nt_steps = 200 # Number of time steps
t = np.linspace(0,t_final, num=nt_steps)
dt = (t_final - t_init)/nt_steps

y0 = [1, -0.1, 0]
training_set = odeint(odemodel,y0,t)
legs = 4 # No. of legs = 1,2,4
slopenet = "LEAPFROG" # Choices: BDF, SEQ, EULER, LEAPFROG
m,n = training_set.shape
n = n+1
xtrain, ytrain = create_training_data(training_set, m, n, dt, legs, slopenet)

# additional data for training with random initial condition
for i in range(-9,11):
    y2s = 0.1*(0.1*i)
    y0 = [1.0, y2s, 0.0]
    training_set = odeint(odemodel, y0, t)
    m,n = training_set.shape
    n = n+1
    xtemp, ytemp = create_training_data(training_set, m, n, dt, legs, slopenet)
    xtrain = np.vstack((xtrain, xtemp))
    ytrain = np.vstack((ytrain, ytemp))

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
x = Dense(100, activation='relu', use_bias=True)(input_layer)
x = Dense(100, activation='relu', use_bias=True)(x)
x = Dense(100, activation='relu', use_bias=True)(x)
x = Dense(100, activation='relu', use_bias=True)(x)
x = Dense(100, activation='relu', use_bias=True)(x)

op_val = Dense(3, activation='linear', use_bias=True)(x)
custom_model = Model(inputs=input_layer, outputs=op_val)
filepath = "best_model.hd5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

custom_model.compile(optimizer='adam', loss='mean_squared_error', metrics=[coeff_determination])
history_callback = custom_model.fit(xtrain, ytrain, epochs=20, batch_size=200, verbose=1, validation_split= 0.3,
                                    callbacks=callbacks_list)

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

y2s = -0.1*(0.25)
y0test = [1, y2s, 0]
testing_set = odeint(odemodel,y0test,t)
m,n = testing_set.shape
n = n+1

#--------------------------------------------------------------------------------------------------------------#
# predict results recursively using the model 
ytest_ml = model_predict(testing_set, m, n, dt, legs, slopenet)

# plot ML prediction and true data
plot_results(ytest_ml, testing_set, m, n)