#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 15:19:23 2019

@author: user1
"""
#namelist = ['SEQ', 'RESNET','EULER', 'BDF2', 'BDF3']
#plist = [1,2,4]
#for name in namelist:
#    for p in plist:
legs = 1 # No. of legs = 1,2,4 
slopenet = "SEQ" # Choices: BDF2, BDF3, BDF4, SEQ, EULER, LEAPFROG, LEAPFROG-FILTER, RESNET
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
from random import uniform
from create_data_v2 import * 
from model_prediction_v2 import *
from export_data_v2 import *
import time as cputime

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
nt_steps = 500 # Number of time steps
t = np.linspace(0,t_final, num=nt_steps)
dt = (t_final - t_init)/nt_steps
sigma = 0.05
y0 = [1, -0.1, 0]

start_ode = cputime.time()

training_set = odeint(odemodel,y0,t)
m,n = training_set.shape
n = n+1

end_ode = cputime.time()
ode_time = end_ode-start_ode

start_train = cputime.time()
xtrain, ytrain = create_training_data(training_set, m, n, dt, legs, slopenet)
a = np.zeros(300)
k = 0
a[k] = -0.1
# additional data for training with random initial condition
#for i in range(-99,101):
#    y2s = 0.1*(0.1*0.1*i)
#    k = k+1
#    a[k] = y2s
#    y0 = [1.0, y2s, 0.0]
#    training_set = odeint(odemodel, y0, t)
#    m,n = training_set.shape
#    n = n+1
#    xtemp, ytemp = create_training_data(training_set, m, n, dt, legs, slopenet)
#    xtrain = np.vstack((xtrain, xtemp))
#    ytrain = np.vstack((ytrain, ytemp))
#
##for i in range(-9,11):
##    y2s = 0.1*(0.001*i)
##    a[k] = y2s
##    k = k+1
##    y0 = [1.0, y2s, 0.0]
##    training_set = odeint(odemodel, y0, t)
##    m,n = training_set.shape
##    n = n+1
##    xtemp, ytemp = create_training_data(training_set, m, n, dt, legs, slopenet)
##    xtrain = np.vstack((xtrain, xtemp))
##    ytrain = np.vstack((ytrain, ytemp))
#    
#indices = np.random.randint(0,xtrain.shape[0],8000)
#xtrain = xtrain[indices]
#ytrain = ytrain[indices]

#%%
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

#%%
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
#model.add(Dropout(0.2))

# Hidden layers
x = Dense(20, activation='tanh', use_bias=True)(input_layer)
x = Dense(20, activation='tanh', use_bias=True)(x)
x = Dense(20, activation='tanh', use_bias=True)(x)
#x = Dense(20, activation='tanh', use_bias=True)(x)
#x = Dense(100, activation='relu', use_bias=True)(x)

op_val = Dense(3, activation='linear', use_bias=True)(x)
custom_model = Model(inputs=input_layer, outputs=op_val)
filepath = "best_model.hd5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

custom_model.compile(optimizer='adam', loss='mean_squared_error', metrics=[coeff_determination])
history_callback = custom_model.fit(xtrain, ytrain, epochs=500, batch_size=50, verbose=1, validation_split= 0.3,
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
#plt.savefig("model4.eps")
plt.show()

#        loss_history = np.array(loss_history)
#        val_loss_history = np.array(val_loss_history)
#        loss_history = loss_history.reshape(500,1)
#        val_loss_history = val_loss_history.reshape(500,1)
#        history = np.hstack((loss_history, val_loss_history))
#        np.savetxt("model4.csv", history, delimiter=",")

#%%
#--------------------------------------------------------------------------------------------------------------#
# predict results recursively using the model
y2s = 0.1*(1)
y0test = [1, y2s, 0]
testing_set1 = odeint(odemodel,y0test,t)
m,n = testing_set1.shape
n = n+1


start_test1 = cputime.time()
ytest_ml1 = model_predict(testing_set1, m, n, dt, legs, slopenet, sigma, sc_input, sc_output)

# sum of L2 norm of each series
l2norm_sum1, l2norm_nd1 = calculate_l2norm(ytest_ml1, testing_set1, m, n, legs, slopenet, problem, y2s)

end_test1 = cputime.time()
test_time1 = end_test1-start_test1
list = [legs, slopenet, problem, train_time, test_time1, train_time+test_time1, ode_time, y2s]
with open('time.csv', 'a') as f:
    f.write("\n")
    for item in list:
        f.write("%s\t" % item)

y2s = -0.1*(1)
y0test = [1, y2s, 0]
testing_set2 = odeint(odemodel,y0test,t)
m,n = testing_set2.shape
n = n+1

start_test2 = cputime.time()
ytest_ml2 = model_predict(testing_set2, m, n, dt, legs, slopenet, sigma,sc_input, sc_output)

# sum of L2 norm of each series
l2norm_sum2, l2norm_nd2 = calculate_l2norm(ytest_ml2, testing_set2, m, n, legs, slopenet, problem, y2s)

end_test2 = cputime.time()
test_time2 = end_test2-start_test2
list = [legs, slopenet, problem, train_time, test_time2, train_time+test_time2, ode_time, y2s]
with open('time.csv', 'a') as f:
    f.write("\n")
    for item in list:
        f.write("%s\t" % item)
        
y2s =0.1*(0.00)
y0test = [1, y2s, 0]
testing_set3 = odeint(odemodel,y0test,t)
m,n = testing_set3.shape
n = n+1

start_test3 = cputime.time()
ytest_ml3 = model_predict(testing_set3, m, n, dt, legs, slopenet, sigma,sc_input, sc_output)

# sum of L2 norm of each series
l2norm_sum3, l2norm_nd3 = calculate_l2norm(ytest_ml3, testing_set3, m, n, legs, slopenet, problem, y2s)

end_test3 = cputime.time()
test_time3 = end_test3-start_test3
list = [legs, slopenet, problem, train_time, test_time3, train_time+test_time3, ode_time, y2s]
with open('time.csv', 'a') as f:
    f.write("\n")
    for item in list:
        f.write("%s\t" % item)
        
# export the solution in .csv file for further post processing
ytest_ml = np.hstack((ytest_ml1, ytest_ml2, ytest_ml3))
testing_set = np.hstack((testing_set1, testing_set2, testing_set3))
export_results_ode(ytest_ml, testing_set, m, n, slopenet, legs)

# plot ML prediction and true data
plot_results_ode(slopenet, legs)
