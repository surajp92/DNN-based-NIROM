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
from matplotlib import pylab as plb
from random import uniform
from create_data_v2 import *
from model_prediction_v2 import *
from export_data_v2 import *


def rk41(_sig, _dt, x, y):
    k1 = _dt*_sig*(y-x)
    k2 = _dt*_sig*(y-(x+k1/2.0))
    k3 = _dt*_sig*(y-(x+k2/2.0))
    k4 = _dt*_sig*(y-(x+k3))
    x = x + k1/6.0 + (k2 + k3)/3.0 + k4/6.0
    return x


def rk42(_r, _dt, x, y, z):
    k1 = _dt*(_r*x-y-x*z)
    k2 = _dt*(_r*x-(y+k1/2.0)-x*z)
    k3 = _dt*(_r*x-(y+k2/2.0)-x*z)
    k4 = _dt*(_r*x-(y+k3)-x*z)
    y = y + k1/6.0 + (k2 + k3)/3.0 + k4/6.0
    return y


def rk43(_k, _dt, x, y, z):
    k1 = _dt*(y*x-_k*z)
    k2 = _dt*(y*x-_k*(z+k1/2.0))
    k3 = _dt*(y*x-_k*(z+k2/2.0))
    k4 = _dt*(y*x-_k*(z+k3))
    z = z + k1/6.0 + (k2 + k3)/3.0 + k4/6.0
    return z


def ode(y0, _t, _dt, _nt_steps):
    a = np.zeros(_t.shape)
    b = np.zeros(_t.shape)
    c = np.zeros(_t.shape)

    sig = 10.0
    k = 8.0/3.0
    r = 23.0

    a[0] = y0[0]
    b[0] = y0[1]
    c[0] = y0[2]

    for i in range(0, _nt_steps):
        a[i+1] = rk41(sig, _dt, a[i], b[i])
        b[i+1] = rk42(r,  _dt, a[i], b[i], c[i])
        c[i+1] = rk43(k,  _dt, a[i], b[i], c[i])
    return a, b, c


# time points
t_init  = 0.0  # Initial time
t_final = 20.0 # Final time
nt_steps = 2000 # Number of time steps
t = np.linspace(0,t_final, num=nt_steps+1)
dt = (t_final - t_init)/nt_steps

y0 = [1, 1, 1]
y1, y2, y3 = ode(y0, t, dt, nt_steps)
training_set = np.vstack([y1,y2,y3]).T
#training_set = states[0:500,:]
 
#plb.figure()
#plb.plot(training_set[:,0],training_set[:,1], '-b')
#plt.xlabel('x')
#plt.ylabel('y')
#plb.figure()
#plb.plot(training_set[:,1],training_set[:,2], '-b')
#plt.xlabel('y')
#plt.ylabel('z')
#plb.figure()
#plb.plot(training_set[:,0],training_set[:,2], '-b')
#plt.xlabel('x')
#plt.ylabel('z')
#plb.show()

legs = 4 # No. of legs = 1,2,4
slopenet = "EULER" # Choices: BDF, SEQ, EULER, LEAPFROG
m,n = training_set.shape
n = n+1
xtrain, ytrain = create_training_data(training_set, m, n, dt, legs, slopenet)

# additional data for training with random initial condition
#for i in range(-9,11):
#    y2s = 0.1*(0.1*i)
#    y0 = [1.0, y2s, 1.0]
#    y1, y2, y3 = ode(y0, t, dt, nt_steps)
#    training_set = np.vstack([y1,y2,y3]).T
#    m,n = training_set.shape
#    n = n+1
#    xtemp, ytemp = create_training_data(training_set, m, n, dt, legs, slopenet)
#    xtrain = np.vstack((xtrain, xtemp))
#    ytrain = np.vstack((ytrain, ytemp))

#indices = np.random.randint(0,xtrain.shape[0],6000)
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
x = Dense(100, activation='relu', use_bias=True)(input_layer)
x = Dense(100, activation='relu', use_bias=True)(x)
#x = Dense(100, activation='relu', use_bias=True)(x)
#x = Dense(100, activation='relu', use_bias=True)(x)
#x = Dense(100, activation='relu', use_bias=True)(x)

op_val = Dense(3, activation='linear', use_bias=True)(x)
custom_model = Model(inputs=input_layer, outputs=op_val)
filepath = "best_model.hd5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

custom_model.compile(optimizer='adam', loss='mean_squared_error', metrics=[coeff_determination])
history_callback = custom_model.fit(xtrain, ytrain, epochs=400, batch_size=200, verbose=1, validation_split= 0.2,
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

y2s = -0.1*(0.25)
y0test = [1., 1., 1.]
y1, y2, y3 = ode(y0test, t, dt, nt_steps)
testing_set = np.vstack([y1,y2,y3]).T
#testing_set = states
m,n = testing_set.shape
n = n+1
sigma = 0.1

#--------------------------------------------------------------------------------------------------------------#
# predict results recursively using the model
ytest_ml = model_predict(testing_set, m, n, dt, legs, slopenet, sigma, sc_input, sc_output)

# plot ML prediction and true data
export_results_ode(ytest_ml, testing_set, m, n, slopenet, legs)

plot_results_lorenzold(slopenet, legs)
