#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import pandas as pd
import time as tm

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import cross_val_score

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import optimizers
# from keras import regularizers

#%%
def create_training_data_lstm(training_set, m, n, lookback):
    ytrain = [training_set[i+1] for i in range(lookback-1,m-1)]
    ytrain = np.array(ytrain)
    xtrain = np.zeros((m-lookback,lookback,n))
    for i in range(m-lookback):
        a = training_set[i]
        for j in range(1,lookback):
            a = np.vstack((a,training_set[i+j]))
        xtrain[i] = a

    return xtrain, ytrain


def export_results_rom(ytest_ml, testing_set, time, m, n, slopenet, legs):
    # export result in x y format for further plotting
    filename = slopenet+'_p=0'+str(legs)+'_'+str(n)+'.csv'
    time = time.reshape(m,1)
    results = np.hstack((time, testing_set, ytest_ml))
    print(filename)
    np.savetxt(filename, results, delimiter=",")

#%%
lookback = 5
problem = "ROM"
slopenet = "LSTM"
legs = lookback

#%% read data for training
dataset_train = pd.read_csv('./a10.csv', sep=",",skiprows=0,header = None, nrows=400)
training_set = dataset_train.iloc[:,1:].values
m,n = training_set.shape

#scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(-1,1))
training_set_scaled = sc.fit_transform(training_set)
training_set = training_set_scaled

#%%
data_sc, labels_sc = create_training_data_lstm(training_set, m, n, lookback)

xtrain, xvalid, ytrain, yvalid = train_test_split(data_sc, labels_sc, test_size=0.1 , shuffle= True)

#%% # create the LSTM model
training_time_init = tm.time()

# keras model
def build_lstm_model(n_blocks=6, n_cells=40, lr=0.001, lookback=5, n=10):
    model = Sequential()
    
    for i in range(n_blocks-1):
        model.add(LSTM(n_cells, input_shape=(lookback, n), return_sequences=True, activation='tanh', kernel_initializer='uniform'))
    
    model.add(LSTM(n_cells, input_shape=(lookback, n), activation='tanh', kernel_initializer='uniform'))
    model.add(Dense(n))
    
    adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mean_squared_error'])
    
    return model

# pass in fixed parameters n_input and n_class
model_lstm = KerasRegressor(
    build_fn = build_lstm_model,
    epochs=10, batch_size=16, verbose=0)

# specify other extra parameters pass to the .fit
# number of epochs is set to a large number
keras_fit_params = {   
    'epochs': 200,
    'batch_size': 16,
    'validation_data': (xvalid, yvalid),
    'verbose': 0
}

# random search parameters 
# specify the options and store them inside the dictionary
# batch size and training method can also be hyperparameters, but it is fixed
n_blocks_params = [3, 4, 5, 6, 7, 8]
n_cells_params = [20, 30, 40, 50, 60]
lr_params = [0.001, 0.0001]

keras_param_options = {
    'n_blocks': n_blocks_params,
    'n_cells': n_cells_params,  
    'lr': lr_params
}

# `verbose` 2 will print the class info for every cross validation, kind of too much
rs_lstm = RandomizedSearchCV( 
    estimator = model_lstm, 
    param_distributions = keras_param_options,
    n_iter = 5,
    scoring = 'neg_mean_squared_error',
    fit_params = keras_fit_params, 
    cv = 5,
    n_jobs = -1,
    verbose = 1
)

rs_result = rs_lstm.fit(xtrain, ytrain)

means = rs_result.cv_results_['mean_test_score']
stds = rs_result.cv_results_['std_test_score']
params = rs_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
print('Best score obtained: {0}'.format(rs_lstm.best_score_))
print('Parameters:')
for param, value in rs_lstm.best_params_.items():
    print('\t{}: {}'.format(param, value))

total_training_time = tm.time() - training_time_init
print('Total training time=', total_training_time)
cpu = open("a_cpu.txt", "w+")
cpu.write('training time in seconds =')
cpu.write(str(total_training_time))
cpu.write('\n')

#%% 
#cross_val_score(model_lstm, xtrain, ytrain, cv=5)
#total_training_time = tm.time() - training_time_init
#print(total_training_time)
    
#grid = GridSearchCV(estimator=model_lstm, param_grid=keras_param_options,scoring = score,
#                    cv = 5, n_jobs=-1)
#
#grid_result = grid.fit(xtrain, ytrain)
#
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
#    print("%f (%f) with: %r" % (mean, stdev, param))

#%%
#read data for testing
dataset_test = pd.read_csv('./a10.csv', sep=",",header = None, skiprows=0)
time = dataset_test.iloc[:,0].values
testing_set = dataset_test.iloc[:,1:].values
m,n = testing_set.shape

testing_set_scaled = sc.fit_transform(testing_set)
testing_set= testing_set_scaled

#%%
m,n = testing_set.shape
ytest = np.zeros((1,lookback,n))
ytest_ml = np.zeros((m,n))

# create input at t = 0 for the model testing
for i in range(lookback):
    ytest[0,i,:] = testing_set[i]
    ytest_ml[i] = testing_set[i]

#%%
testing_time_init = tm.time()

# predict results recursively using the model
for i in range(lookback,m):
    slope_ml = model.predict(ytest)
    ytest_ml[i] = slope_ml
    e = ytest
    for i in range(lookback-1):
        e[0,i,:] = e[0,i+1,:]
    e[0,lookback-1,:] = slope_ml
    ytest = e 

#%%
total_testing_time = tm.time() - testing_time_init
print('Total testing time=', total_testing_time)
cpu.write('testing time in seconds = ')
cpu.write(str(total_testing_time))
cpu.close()

#%%  unscaling
ytest_ml_unscaled = sc.inverse_transform(ytest_ml)
ytest_ml= ytest_ml_unscaled

testing_set_unscaled = sc.inverse_transform(testing_set)
testing_set= testing_set_unscaled

# export the solution in .csv file for further post processing
export_results_rom(ytest_ml, testing_set, time, m, n, slopenet, legs)

time = time.reshape(m,1)
filename = 'aml.csv'
rt = np.hstack((time, ytest_ml))
np.savetxt(filename, rt, delimiter=",")

#%%
at = np.loadtxt(open('a10.csv', "rb"), delimiter=",", skiprows=0)
aml = np.loadtxt(open('aml.csv', "rb"), delimiter=",", skiprows=0)
m,n = at.shape

# plotting time series
plt.rc('text', usetex=True)
plt.rc('font', family='Times New Roman')

nrows = int(n/2)
k = 1
fig, axs = plt.subplots(nrows, 2, figsize=(13,12))
if nrows == 1:  # for 2 modes training
    for j in range(2):
        axs[j].plot(at[:,0],at[:,k], color='black', linestyle='-', label=r'$y_'+str(1)+'$'+' (True)', zorder=5)
        axs[j].plot(at[:401,0],aml[:401,k], color='orange', linestyle='--', label=r'$y_'+str(i+1)+'$'+' (ML-Train)', zorder=5)
        axs[j].plot(at[401:,0],aml[401:,k], color='red', linestyle='--', label=r'$y_'+str(i+1)+'$'+' (ML-Test)', zorder=5)
        axs[j].set_xlim([10., at[m-1,0]])
        axs[j].set_ylim([-250., 250.])
        axs[j].set_ylabel('$a_'+'{'+(str(k)+'}'+'$'), labelpad = -6, fontsize = 14)
        k = int(k+1)

else:
    for i in range(nrows):  # for more than 2 modes training
        for j in range(2):
            axs[i,j].plot(at[:,0],at[:,k], color='black', linestyle='-', label=r'$y_'+str(i+1)+'$'+' (True)', zorder=5)
            axs[i,j].plot(at[:401,0],aml[:401,k], color='darkorange', linestyle='--', label=r'$y_'+str(i+1)+'$'+' (ML-Train)', zorder=5)
            axs[i,j].plot(at[401:,0],aml[401:,k], color='r', linestyle='--', label=r'$y_'+str(i+1)+'$'+' (ML-Test)', zorder=5)
            axs[i,j].set_xlim([10., at[m-1,0]])
            axs[i,j].set_ylim([-250., 250.])
            axs[i,j].set_ylabel('$a_'+'{'+(str(k)+'}'+'(t)$'), labelpad = -6, fontsize = 14)
            k = int(k+1)

fig.tight_layout()
axs[nrows-1, 0].set_xlabel('Time', labelpad = 10, fontsize = 14)
axs[nrows-1, 1].set_xlabel('Time', labelpad = 10, fontsize = 14)
fig.subplots_adjust(bottom=0.1)

line_labels = ["True", "ML-Train", "ML-Test"]
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.1, ncol=6, labelspacing=0.,  prop={'size': 13})

# save figure
plt.savefig("lstm10.eps", bbox_inches = 'tight', dpi=200)
plt.savefig("lstm10.pdf", bbox_inches='tight', dpi=200)

