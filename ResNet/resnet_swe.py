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

#%%
legs = 4# No. of legs = 1,2,4 
slopenet = 'RESNET' # Choices: BDF2, SEQ, EULER, RESNET
problem = "SWE"

import os
if os.path.isfile('best_model.hd5'):
    os.remove('best_model.hd5')

#%%
import numpy as np
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import matplotlib.pyplot as plt
from create_data import *
from model_prediction import *
from export_data import *
import time as cputime
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import optimizers
from keras import losses
from keras import regularizers

#%%
def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


def customloss(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

#%%
dataset_train = np.genfromtxt('./DNS_c_4.csv', delimiter=",",skip_header=0)
training_set = dataset_train[:1441,1:5]
m,n=training_set.shape
t = dataset_train[:1441,0]
dt = t[1] - t[0]

start_train = cputime.time()

xtrain, ytrain = create_training_data(training_set, dt, legs, slopenet)

#%%
from sklearn.preprocessing import MinMaxScaler
sc_input = MinMaxScaler(feature_range=(-1,1))
sc_input = sc_input.fit(xtrain)
xtrain_scaled = sc_input.transform(xtrain)
xtrain = xtrain_scaled

from sklearn.preprocessing import MinMaxScaler
sc_output = MinMaxScaler(feature_range=(-1,1))
sc_output = sc_output.fit(ytrain)
ytrain_scaled = sc_output.transform(ytrain)
ytrain = ytrain_scaled

indices = np.random.randint(0,xtrain.shape[0],1000)
xtrain = xtrain[indices]
ytrain = ytrain[indices]

#%%
model = Sequential()

# Layers start
input_layer = Input(shape=(legs*n,))
#model.add(Dropout(0.2))

# Hidden layers
x = Dense(80, activation='tanh',  kernel_initializer='glorot_normal', use_bias=True)(input_layer)
x = Dense(80, activation='tanh',  kernel_initializer='glorot_normal', use_bias=True)(x)
x = Dense(80, activation='tanh',  kernel_initializer='glorot_normal', use_bias=True)(x)
x = Dense(80, activation='tanh',  kernel_initializer='glorot_normal', use_bias=True)(x)

#x = Dense(240, activation='relu',  use_bias=True)(x)
#x = Dense(240, activation='relu',  use_bias=True)(x)

op_val = Dense(n, activation='linear', use_bias=True)(x)
custom_model = Model(inputs=input_layer, outputs=op_val)
filepath = "best_model.hd5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

custom_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
history_callback = custom_model.fit(xtrain, ytrain, epochs=900, batch_size=120, verbose=1, validation_split= 0.1,
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
plt.show()

#%% read data for testing
dataset_train = np.genfromtxt('./DNS_c_4.csv', delimiter=",",skip_header=0)
testing_set = dataset_train[:1441,1:5]
m,n = testing_set.shape
custom_model = load_model('best_model.hd5')#,custom_objects={'coeff_determination': coeff_determination})

#%%
ytest_ml = model_predict(testing_set, dt, legs, slopenet, sc_input, sc_output)

#%%
# sum of L2 norm of each series
l2norm_sum, l2norm_nd = calculate_l2norm(ytest_ml, testing_set,legs, slopenet, problem)

# export the solution in .csv file for further post processing
export_results(ytest_ml, testing_set, t, slopenet, legs)

#%%
# plot ML prediction and true data
#plot_results_swe(dt, slopenet, legs)

font = {'family' : 'Times New Roman',
        'size'   : 14}	
plt.rc('font', **font)
nrows = 4
#list = [0,3,7,11,15,19,23,27,31]
list = [0,1,2,3,4]

fig, axs = plt.subplots(nrows, 1, figsize=(10,8))#, constrained_layout=True)
for i,j in zip(range(nrows),list):
    #axs[i].plot(ytrains[:,i], color='black', linestyle='-', label=r'$y_'+str(i+1)+'$'+' (True)', zorder=5)
    axs[i].plot(t,testing_set[:,j], color='black', linestyle='-', label=r'$y_'+str(i+1)+'$'+' (True)', zorder=5)
    #axs[i].plot(t,gp[:,j], color='blue', linestyle='-', label=r'$y_'+str(i+1)+'$'+' (True)', zorder=5)
    axs[i].plot(t,ytest_ml[:,j], color='red', linestyle='-', label=r'$y_'+str(i+1)+'$'+' (GP)', zorder=5) 
    axs[i].set_xlim([t[0], t[m-1]])
    #axs[i].set_ylim([min(ytest_ml[:,j])-0.2, max(ytest_ml[:,j])+0.2])
    #axs[i].set_ylim([-(max(ytest_ml[:,j])*1.25), (max(ytest_ml[:,j])*1.25)])
    axs[i].set_ylabel('$c_'+'{'+(str(j+1)+'}'+'$'), fontsize = 14)
    axs[i].axvspan(t[0], t[0]+0.5*(t[m-1]-t[0]), facecolor='0.5', alpha=0.5)
   
fig.tight_layout() 

#fig.subplots_adjust(bottom=0.15)

line_labels = ["True", "GP", "ML"]#, "ML-Train", "ML-Test"]
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.1, ncol=6, labelspacing=0.,  prop={'size': 13} ) #bbox_to_anchor=(0.5, 0.0), borderaxespad=0.1, 

savefile = slopenet+'_p=0'+str(legs)+'.eps'
fig.savefig(savefile)#, bbox_inches = 'tight', pad_inches = 0.01)