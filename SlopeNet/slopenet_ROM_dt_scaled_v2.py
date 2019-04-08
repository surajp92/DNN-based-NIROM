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
dataset_train = pd.read_csv('./a.csv', sep=",",skiprows=0,header = None, nrows=1000)
m,n=dataset_train.shape
training_set = dataset_train.iloc[:,0:n].values
time = training_set[:,0]
dt = training_set[1,0] - training_set[0,0]
training_set = training_set[:,1:n]

#from sklearn.preprocessing import MinMaxScaler
#sc = MinMaxScaler(feature_range=(0,1))
#training_set_scaled = sc.fit_transform(training_set)
#training_set_scaled.shape
#training_set = training_set_scaled

# arrange trainig data
dt_scaling = 4.0
indices = [int(dt_scaling*i) for i in range(int(m/dt_scaling))]
set1 = training_set[indices]
training_set = set1
dt = dt*dt_scaling
m,n = training_set.shape
n = n+1

legs = 4 # No. of legs = 1,2,4 
slopenet = "EULER" # Choices: BDF2, BDF3, BDF4, SEQ, EULER, LEAPFROG, LEAPFROG-FILTER
sigma = 0.06
problem = "ROM"

start_train = cputime.time()

xtrain, ytrain = create_training_data(training_set, m, n, dt, legs, slopenet)

#indices = np.random.randint(0,xtrain.shape[0],500)
#xtrain = xtrain[indices]
#ytrain = ytrain[indices]

#--------------------------------------------------------------------------------------------------------------#
# create the LSTM model
model = Sequential()

# Layers start
input_layer = Input(shape=(legs*(n-1),))
#model.add(Dropout(0.2))

# Hidden layers
x = Dense(100, activation='tanh', use_bias=True)(input_layer)
x = Dense(100, activation='tanh', use_bias=True)(x)
#x = Dense(400, activation='relu', use_bias=True)(x)
#x = Dense(400, activation='relu', use_bias=True)(x)
#x = Dense(400, activation='relu', use_bias=True)(x)
#x = Dense(400, activation='relu', use_bias=True)(x)
#x = Dense(400, activation='relu', use_bias=True)(x)


op_val = Dense(n-1, activation='linear', use_bias=True)(x)
custom_model = Model(inputs=input_layer, outputs=op_val)
filepath = "best_model.hd5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

custom_model.compile(optimizer='adam', loss='mean_squared_error', metrics=[coeff_determination])
history_callback = custom_model.fit(xtrain, ytrain, epochs=900, batch_size=90, verbose=1, validation_split= 0.1,
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
plt.plot(epochs, loss_history, 'b', label='Training loss')
plt.plot(epochs, val_loss_history, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#%%
#--------------------------------------------------------------------------------------------------------------#
#read data for testing
dataset_test = pd.read_csv('./a.csv', sep=",",header = None, skiprows=0, nrows=2000)
dataset_total = pd.concat((dataset_train,dataset_test),axis=0)
dataset_total.drop(dataset_total.columns[[0]], axis=1, inplace=True)
m,n=dataset_test.shape
testing_set = dataset_test.iloc[:,0:n].values
time = testing_set[:,0]
testing_set = testing_set[:,1:n]
indices_test = [int(dt_scaling*i) for i in range(int(m/dt_scaling))]

#%%
#testing_set_scaled = sc.fit_transform(testing_set)
#testing_set_scaled.shape
#testing_set= testing_set_scaled

#--------------------------------------------------------------------------------------------------------------#
# predict results recursively using the model 
start_test = cputime.time()
dt1 = dt
m = int(m/dt_scaling)
ytest_ml = model_predict(testing_set, m, n, dt1, legs, slopenet, sigma)

end_test = cputime.time()
test_time = end_test - start_test

list = [legs, slopenet, problem, train_time, test_time, train_time+test_time]
with open('time.csv', 'a') as f:
    f.write("\n")
    for item in list:
        f.write("%s\t" % item)

#ytest_ml_unscaled = sc.inverse_transform(ytest_ml)
#ytest_ml_unscaled.shape
#ytest_ml= ytest_ml_unscaled
#
#testing_set_unscaled = sc.inverse_transform(testing_set)
#testing_set_unscaled.shape
#testing_set= testing_set_unscaled

time = time-900
time_test = time.reshape(int(dt_scaling*m),1)
testing_set = np.hstack((time_test, testing_set))
time_mlpred = time[indices_test]
time_mlpred = time_mlpred.reshape(m,1)
ytest_ml = np.hstack((time_mlpred, ytest_ml))

np.savetxt("testing_data.csv", testing_set, delimiter=",")
np.savetxt("ml_solution.csv", ytest_ml, delimiter=",")

#plotting
ytest_mltrain = ytest_ml[0:int(m/2.0)]
ytest_mlpred = ytest_ml[int(m/2.0):]

#%%
for i in range(1,n):
    plt.figure()    
    plt.plot(testing_set[:,0],testing_set[:,i], 'r-', label=r'$y_'+str(i+1)+'$'+' (True)')
    plt.plot(ytest_mltrain[:,0],ytest_mltrain[:,i], 'b-', label=r'$y_'+str(i+1)+'$'+' (ML Test)')
    plt.plot(ytest_mlpred[:,0],ytest_mlpred[:,i], 'g-', label=r'$y_'+str(1+1)+'$'+' (ML Pred)')
    plt.ylabel('Response')
    plt.xlabel('Time')
    plt.legend(loc='best')
    name = 'a='+str(i+1)+'.eps'
        #plt.savefig(name, dpi = 400)
    plt.show()

#%%    
# sum of L2 norm of each series
testing_set_l2norm = testing_set[indices_test]
testing_set_l2norm = testing_set_l2norm[:,1:] 
l2norm_sum, l2norm_nd = calculate_l2norm(ytest_ml[:,1:], testing_set_l2norm, m, n, legs, slopenet, problem)
