#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 13:38:12 2019

@author: Suraj Pawar
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
from scipy.integrate import odeint

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras import initializers
import keras.backend as K

#%%
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
dt = 0.05
t = np.arange(t_init, t_final, dt)
states = odeint(f, state0, t)
nsamples = int((t_final-t_init)/dt)

#%%
training_set = states[0:nsamples:] # training data between t_init and (t_final/2)
t = t.reshape(nsamples,1)
training_set = np.hstack((t, training_set))
m,n = training_set.shape
lookback = 4   # history for next time step prediction 
slopenet = "LSTM"

def customloss(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

def create_training_data_lstm(training_set, m, n, dt, lookback):
    """ 
    This function creates the input and output of the neural network. The function takes the
    training data as an input and creates the input for LSTM neural network based on the lookback.
    The input for the LSTM is 3-dimensional matrix.
    For lookback = 2;  [y(n-2), y(n-1)] ------> [y(n)]
    """
    ytrain = [training_set[i+1,1:] for i in range(lookback-1,m-1)]
    ytrain = np.array(ytrain)
    xtrain = np.zeros((m-lookback,lookback,n))
    for i in range(m-lookback):
        a = training_set[i]
        for j in range(1,lookback):
            a = np.vstack((a,training_set[i+j]))
        xtrain[i] = a
    
    return xtrain, ytrain

#%%
xtrain, ytrain = create_training_data_lstm(training_set, m, n, dt, lookback)

#%% calculate maximum and minimum for scaling
xmax, xmin = [np.zeros(n) for i in range(2)]
ymax, ymin = [np.zeros(n-1) for i in range(2)]
for i in range(n):
    xmax[i] = max(training_set[:,i])
    xmin[i] = min(training_set[:,i])

for i in range(n-1):
    ymax[i] = max(ytrain[:,i])
    ymin[i] = min(ytrain[:,i])

#%% scale the input and output data to (-1,1)
xtrains = np.zeros((m-lookback,lookback,n)) 
ytrains = np.zeros((m-lookback,n-1)) 
for i in range(n):
    xtrains[:,:,i] = (2.0*xtrain[:,:,i]-(xmax[i]+xmin[i]))/(xmax[i]-xmin[i])
    
for i in range(n-1):
    ytrains[:,i] = (2.0*ytrain[:,i]-(ymax[i]+ymin[i]))/(ymax[i]-ymin[i])
    
#%%    
model = Sequential()

model.add(LSTM(40, input_shape=(lookback, n), return_sequences=True, activation='tanh', kernel_initializer='uniform'))
model.add(LSTM(40, input_shape=(lookback, n), return_sequences=True, activation='tanh', kernel_initializer='uniform'))
model.add(LSTM(40, input_shape=(lookback, n), return_sequences=True, activation='tanh', kernel_initializer='uniform'))
model.add(LSTM(40, input_shape=(lookback, n), return_sequences=True, activation='tanh', kernel_initializer='uniform'))
model.add(LSTM(40, input_shape=(lookback, n), activation='tanh', kernel_initializer='uniform'))
model.add(Dense(n-1))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy']) # compile the model

history = model.fit(xtrains, ytrains, nb_epoch=400, batch_size=50, validation_split=0.2) # run the model

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.figure()
plt.semilogy(epochs, loss, 'b', label='Training loss')
plt.semilogy(epochs, val_loss, 'r', label='Validationloss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#%%
testing_set = states
testing_set = np.hstack((t, testing_set))
m,n = testing_set.shape

#%%
xt = np.zeros((1,lookback,n))
yt_ml = np.zeros((m,n-1))
xt_check = np.zeros((m,lookback,n))

for i in range(lookback):
    # create data for testing at first time step
    xt[0,i,:] = testing_set[i]
    yt_ml[i] = testing_set[i,1:]

xt_check[0,:,:] = xt
#%%
xt_sc = np.zeros((1,lookback,n))
yt_sc = np.zeros((1,n-1))

for i in range(lookback,m):
    for k in range(n):
        xt_sc[:,:,k] = (2.0*xt[:,:,k]-(xmax[k]+xmin[k]))/(xmax[k]-xmin[k])
    yt = model.predict(xt_sc) # slope from LSTM/ ML model
    for k in range(n-1):
        yt_sc[0,k] = 0.5*(yt[0,k]*(ymax[k]-ymin[k])+(ymax[k]+ymin[k]))
    yt_ml[i] = yt_sc # assign variable at next time ste y(n)
    e = xt.copy()   # temporaty variable
    for j in range(lookback-1):
        e[0,j,:] = e[0,j+1,:]
    e[0,lookback-1,0] = testing_set[i,0]
    e[0,lookback-1,1:] = yt_sc
    xt = e # update the input for the variable prediction at time step (n+1)
    xt_check[i,:,:] = xt

ytest_ml = yt_ml    

#%%
filename = slopenet+'_p=0'+str(lookback)+'.csv'
results = np.hstack((testing_set, ytest_ml))
np.savetxt(filename, results, delimiter=",")
    
filename = slopenet+'_p=0'+str(lookback)+'.csv'
solution = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=0)
m,n = solution.shape
time = solution[:,0]
ytrue = solution[:,1:int((n-1)/2+1)]

ytestplot = solution[0:int(m/2),int((n-1)/2+1):]
ypredplot = solution[int(m/2):,int((n-1)/2+1):]
for i in range(int(n/2)):
    plt.figure()    
    plt.plot(time,ytrue[:,i], 'r-', label=r'$y_'+str(i+1)+'$'+' (True)')
    plt.plot(time[0:int(m/2)],ytestplot[:,i], 'b-', label=r'$y_'+str(i+1)+'$'+' (ML Test)')
    plt.plot(time[int(m/2):],ypredplot[:,i], 'g-', label=r'$y_'+str(1+1)+'$'+' (ML Pred)')
    plt.ylabel('Response')
    plt.xlabel('Time')
    plt.legend(loc='best')
    plt.show()




