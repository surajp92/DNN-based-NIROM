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

# additional data for training with random initial condition
for i in range(-9,11):
    y2s = 0.1*(0.1*i)
    y0 = [1.0, y2s, 0.0]
    ytemp = odeint(odemodel, y0, t)
    training_set = np.vstack((training_set, ytemp))

m,n = training_set.shape
n = n+1

legs = 1 # No. of legs = 1,2,4
slopnet = "EULER" # Choices: BDF, SEQ, EULER

xtrain, ytrain = create_training_data(training_set, m, n, dt, legs, slopnet)

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
history_callback = custom_model.fit(xtrain, ytrain, epochs=200, batch_size=200, verbose=1, validation_split= 0.3,
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


nsamples = 1
rmshist = np.zeros((nsamples,3))
y20hist = np.zeros(nsamples)
for k in range(nsamples):
    # initial condition
    #y2s = 0.1*uniform(-1,1)
    y2s = -0.1*(0.25)
    y20hist[k] = y2s
    # initial condition
    y0test = [1, y2s, 0]
    
    # solve ODE
    ytode = odeint(odemodel,y0test,t)
    
    # create input at t= 0 for the model testing
    ytest = [ytode[0]]
    ytest = np.array(ytest)
    # ytest = ytest.reshape(1,1,3)

    # create an array to store ml predicted y
    ytest_ml = np.zeros((nt_steps,3))
    # ytest_ml = ode solution for first four time steps
    ytest_ml[0] = ytode[0]
    
    for i in range(1,nt_steps):
        slope_ml = custom_model.predict(ytest) # slope from LSTM/ ML model
        a = ytest_ml[i-1,0] + 1.0*dt*slope_ml[i-1,0] # y1 at next time step
        b = ytest_ml[i-1,1] + 1.0*dt*slope_ml[i-1,1] # y2 at next time step
        c = ytest_ml[i-1,2] + 1.0*dt*slope_ml[i-1,2] # y3 at next time step
        d = [a,b,c] # [y1, y2, y3] at (n+1)th step
        d = np.array(d)
        ytest_ml[i] = d
        ytest = np.vstack((ytest, d)) # add [y1, y2, y3] at (n+1) to input test for next slope prediction
    
    plt.figure()    
    plt.plot(t, ytest_ml[:,0], 'c-', label=r'$y_1 ML$') 
    plt.plot(t, ytest_ml[:,1], 'm-', label=r'$y_2 ML$') 
    plt.plot(t, ytest_ml[:,2], 'y-', label=r'$y_3 ML$') 
    
    plt.plot(t, ytode[:,0], 'r-', label=r'$y_1$') 
    plt.plot(t, ytode[:,1], 'g-', label=r'$y_2$') 
    plt.plot(t, ytode[:,2], 'b-', label=r'$y_3$') 
    
    plt.ylabel('response ML')
    plt.xlabel('time ML')
    plt.legend(loc='best')
    plt.savefig('ode_euler_oneleg_x=0.25.png', dpi = 1000)
    plt.show()
    
    # calculation of mean square error
    residual = np.zeros((nsamples,3))
    residual = ytest_ml - ytode
    residual2 = residual*residual
    rms = np.zeros((1,3))
    rms = sum(residual2)
    rms = np.sqrt(rms)
    rms = rms/nsamples
    
    rmshist[k] = rms.reshape(1,3)

y20hist = y20hist.reshape(nsamples,1)
errorhist =  np.concatenate((y20hist,rmshist), axis = 1)
errorhist = errorhist[errorhist[:,0].argsort()]

#plt.figure()    
#plt.plot(errorhist[:,0], errorhist[:,1], 'r-', label='Y1 Error') 
#plt.plot(errorhist[:,0], errorhist[:,2], 'g-', label='Y2 Error') 
#plt.plot(errorhist[:,0], errorhist[:,3], 'b-', label='Y3 Error') 
#
#plt.ylabel('Mean square error')
#plt.xlabel('x')
#plt.legend(loc='best')
#plt.show()