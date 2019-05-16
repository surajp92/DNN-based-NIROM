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

legs = 4 # No. of legs = 1,2,4 
slopenet = 'RESNET' # Choices: BDF2, BDF3, BDF4, SEQ, EULER, LEAPFROG, LEAPFROG-FILTER, RESNET
sigma = 0.00
problem = "ROM"

import os
if os.path.isfile('best_model.hd5'):
    os.remove('best_model.hd5')

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
import time as cputime
#--------------------------------------------------------------------------------------------------------------#
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import optimizers
from keras import losses

#--------------------------------------------------------------------------------------------------------------#
def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

#--------------------------------------------------------------------------------------------------------------#
def customloss(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

#--------------------------------------------------------------------------------------------------------------#
dataset_train = pd.read_csv('./a94.csv', sep=",",skiprows=0,header = None, nrows=1000)
m,n=dataset_train.shape
training_set = dataset_train.iloc[:,0:n].values
time = training_set[:,0]
dt = training_set[1,0] - training_set[0,0]
training_set = training_set[:,1:n]

start_train = cputime.time()

xtrain, ytrain = create_training_data(training_set, m, n, dt, legs, slopenet)

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

indices = np.random.randint(0,xtrain.shape[0],1000)
xtrain = xtrain[indices]
ytrain = ytrain[indices]

#--------------------------------------------------------------------------------------------------------------#
# create the LSTM model
model = Sequential()

# Layers start
input_layer = Input(shape=(legs*(n-1),))
#model.add(Dropout(0.2))

# Hidden layers
x = Dense(120, activation='relu', use_bias=True)(input_layer)
x = Dense(120, activation='relu', use_bias=True)(x)
x = Dense(120, activation='relu', use_bias=True)(x)
x = Dense(120, activation='relu', use_bias=True)(x)
x = Dense(120, activation='relu', use_bias=True)(x)
x = Dense(120, activation='relu', use_bias=True)(x)

op_val = Dense(n-1, activation='linear', use_bias=True)(x)
custom_model = Model(inputs=input_layer, outputs=op_val)
filepath = "best_model.hd5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

custom_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
history_callback = custom_model.fit(xtrain, ytrain, epochs=900, batch_size=40, verbose=1, validation_split= 0.2,
                                    callbacks=callbacks_list)

#--------------------------------------------------------------------------------------------------------------#
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
plt.show()

#--------------------------------------------------------------------------------------------------------------#
#read data for testing
dataset_test = pd.read_csv('./a94.csv', sep=",",header = None, skiprows=0)#, nrows=2000)
dataset_total = pd.concat((dataset_train,dataset_test),axis=0)
dataset_total.drop(dataset_total.columns[[0]], axis=1, inplace=True)
m,n=dataset_test.shape
testing_set = dataset_test.iloc[:,0:n].values
time = testing_set[:,0]
testing_set = testing_set[:,1:n]
m = m

#%%

print("rn4")
custom_model = load_model('best_model.hd5')#,custom_objects={'coeff_determination': coeff_determination})

ytest = [testing_set[0], testing_set[1], testing_set[2], testing_set[3]]
ytest = np.array(ytest)
ytest = ytest.reshape(1,4*(n-1))

ytest_ml = [testing_set[0]]
ytest_ml = np.array(ytest_ml)
ytest_ml = np.vstack((ytest_ml, testing_set[1]))
ytest_ml = np.vstack((ytest_ml, testing_set[2]))
ytest_ml = np.vstack((ytest_ml, testing_set[3]))

#%%
for i in range(4, m):
    ytest_sc = sc_input.transform(ytest)
    slope_ml = custom_model.predict(ytest_sc) # slope from LSTM/ ML model
    slope_ml_sc = sc_output.inverse_transform(slope_ml)
    a = ytest_ml[i-1] + slope_ml_sc[0] # y1 at next time step
    ytest_ml = np.vstack((ytest_ml, a))
    e = a.reshape(1,n-1)
    ytemp = ytest[0]
    ytemp = ytemp.reshape(1,4*(n-1))
    ee = np.concatenate((ytemp,e), axis = 1)
    ee = ee[0,n-1:]
    ytest = ee.reshape(1,4*(n-1))

#%%
#--------------------------------------------------------------------------------------------------------------#
# predict results recursively using the model 
start_test = cputime.time()
ytest_ml = model_predict(testing_set, m, n, dt, legs, slopenet, sigma, sc_input, sc_output)

#%%
font = {'family' : 'Times New Roman',
        'size'   : 14}	
plt.rc('font', **font)

nrows = 4
k = 1
t = time

fig, axs = plt.subplots(nrows, 1, figsize=(11,5))#, constrained_layout=True)
for i in range(nrows):
    #axs[i].plot(ytrains[:,i], color='black', linestyle='-', label=r'$y_'+str(i+1)+'$'+' (True)', zorder=5)
    axs[i].plot(t,testing_set[:,i], color='black', linestyle='-', label=r'$y_'+str(i+1)+'$'+' (True)', zorder=5)
    axs[i].plot(t,ytest_ml[:,i], color='blue', linestyle='-.', label=r'$y_'+str(i+1)+'$'+' (GP)', zorder=5) 
    #axs[i].set_xlim([t[0], t[m-1]])
    axs[i].set_ylabel('$a_'+'{'+(str(k)+'}'+'$'), fontsize = 14)
    k = int(k+1)

fig.tight_layout() 

fig.subplots_adjust(bottom=0.1)

line_labels = ["True", "ML"]#, "ML-Train", "ML-Test"]
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.1, ncol=6, labelspacing=0.,  prop={'size': 13} ) #bbox_to_anchor=(0.5, 0.0), borderaxespad=0.1, 

fig.savefig('gp.eps')#, bbox_inches = 'tight', pad_inches = 0.01)

#%%
# sum of L2 norm of each series
l2norm_sum, l2norm_nd = calculate_l2norm(ytest_ml, testing_set, m, n, legs, slopenet, problem)

end_test = cputime.time()
test_time = end_test - start_test

list = [legs, slopenet, problem, train_time, test_time, train_time+test_time]
with open('time.csv', 'a') as f:
    f.write("\n")
    for item in list:
        f.write("%s\t" % item)

# export the solution in .csv file for further post processing
export_results_rom(ytest_ml, testing_set, time, m, n, slopenet, legs)

# plot ML prediction and true data
plot_results_rom(dt,slopenet, legs)

