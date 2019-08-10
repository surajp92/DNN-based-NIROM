#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import pandas as pd
import time as tm

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
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

#%%
def export_results_rom(ytest_ml, testing_set, time, m, n, slopenet, legs):
    # export result in x y format for further plotting
    filename = slopenet+'_p=0'+str(legs)+'.csv'
    time = time.reshape(m,1)
    results = np.hstack((time, testing_set, ytest_ml))
    np.savetxt(filename, results, delimiter=",")

#%%
def plot_results_rom(dt1, slopenet, legs):
    filename = slopenet+'_p=0'+str(legs)+'.csv'
    solution = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=0)
    m,n = solution.shape
    time = solution[:,0]-900
    dt = time[1] - time[0]
    time_test = solution[0:1000,0]-900
    time_pred = solution[1000:,0]-900
    ytrue = solution[:,1:int((n-1)/2+1)]
    ytestplot = solution[0:1000,int((n-1)/2+1):]
    ypredplot = solution[1000:,int((n-1)/2+1):]
    time_test = time[0:1000]
    time_pred = time[1000:]
    for i in range(int(n/2)):
        plt.figure()
        plt.plot(time,ytrue[:,i], 'r-', label=r'$y_'+str(i+1)+'$'+' (True)')
        plt.plot(time[0:1000],ytestplot[:,i], 'b-', label=r'$y_'+str(i+1)+'$'+' (ML Test)')
        plt.plot(time[1000:],ypredplot[:,i], 'g-', label=r'$y_'+str(1+1)+'$'+' (ML Pred)')
        #plt.plot(solution[:,i+int(n/2)], 'r-', label=r'$y_'+str(i+1)+'$'+' (ML)')
        plt.ylabel('Response')
        plt.xlabel('Time')
        plt.legend(loc='best')
        name = 'a='+str(i+1)+'.eps'
        #plt.savefig(name, dpi = 400)
        #plt.show()

#%%
dataset_train = pd.read_csv('./a.csv', sep=",",skiprows=0,header = None, nrows=400)
m,n=dataset_train.shape
training_set = dataset_train.iloc[:,0:n].values
dt = training_set[1,0] - training_set[0,0]
training_set = training_set[:,1:n]
n = n-1
lookback = 5
problem = "ROM"

slopenet = "LSTM"
legs = lookback

#scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(-1,1))
training_set_scaled = sc.fit_transform(training_set)
training_set_scaled.shape
training_set = training_set_scaled

xtrain, ytrain = create_training_data_lstm(training_set, m, n, lookback)

indices = np.random.randint(0, xtrain.shape[0], 400)
xtrain = xtrain[indices]
ytrain = ytrain[indices]

training_time_init = tm.time()

# create the LSTM model
model = Sequential()

model.add(LSTM(40, input_shape=(lookback, n), return_sequences=True, activation='tanh', kernel_initializer='glorot_normal'))
model.add(LSTM(40, input_shape=(lookback, n), return_sequences=True, activation='tanh', kernel_initializer='uniform'))
model.add(LSTM(40, input_shape=(lookback, n), return_sequences=True, activation='tanh', kernel_initializer='uniform'))
model.add(LSTM(40, input_shape=(lookback, n), return_sequences=True, activation='tanh', kernel_initializer='uniform'))
model.add(LSTM(40, input_shape=(lookback, n), return_sequences=True, activation='tanh', kernel_initializer='uniform'))
model.add(LSTM(40, input_shape=(lookback, n), activation='tanh', kernel_initializer='uniform'))
model.add(Dense(n))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

history = model.fit(xtrain, ytrain, nb_epoch=500, batch_size=16, validation_split=0.2)
total_training_time = tm.time() - training_time_init
print('Total training time=', total_training_time)
cpu = open("a_cpu.txt", "w+")
cpu.write('training time in seconds =')
cpu.write(str(total_training_time))
cpu.write('\n')

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
#--------------------------------------------------------------------------------------------------------------#
#read data for testing
dataset_test = pd.read_csv('./a.csv', sep=",",header = None, skiprows=0)
dataset_total = pd.concat((dataset_train,dataset_test),axis=0)
dataset_total.drop(dataset_total.columns[[0]], axis=1, inplace=True)
m,n=dataset_test.shape
testing_set = dataset_test.iloc[:,0:n].values
time = testing_set[:,0]
testing_set = testing_set[:,1:n]

testing_set_scaled = sc.fit_transform(testing_set)
testing_set_scaled.shape
testing_set= testing_set_scaled

m,n = testing_set.shape
ytest = np.zeros((1,lookback,n))
ytest_ml = np.zeros((m,n))

# create input at t = 0 for the model testing
for i in range(lookback):
    ytest[0,i,:] = testing_set[i]
    ytest_ml[i] = testing_set[i]

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

total_testing_time = tm.time() - testing_time_init
print('Total testing time=', total_testing_time)
cpu.write('testing time in seconds = ')
cpu.write(str(total_testing_time))
cpu.close()

n = n+1

#%%
# unscaling
ytest_ml_unscaled = sc.inverse_transform(ytest_ml)
ytest_ml_unscaled.shape
ytest_ml= ytest_ml_unscaled

testing_set_unscaled = sc.inverse_transform(testing_set)
testing_set_unscaled.shape
testing_set= testing_set_unscaled

# export the solution in .csv file for further post processing
export_results_rom(ytest_ml, testing_set, time, m, n, slopenet, legs)

# plot ML prediction and true data
plot_results_rom(dt, slopenet, legs)

#%%
t = time.reshape(m,1)
filename = 'aml.csv'
rt = np.hstack((t, ytest_ml))
np.savetxt(filename, rt, delimiter=",")

#%%
at = np.loadtxt(open('a.csv', "rb"), delimiter=",", skiprows=0)
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
plt.savefig("lstm10.eps", bbox_inches = 'tight', dpi=200)
plt.savefig("lstm10.pdf", bbox_inches='tight', dpi=200)

