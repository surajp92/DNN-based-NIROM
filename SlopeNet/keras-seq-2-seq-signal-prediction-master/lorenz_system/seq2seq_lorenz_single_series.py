#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 10:56:34 2019

@author: Suraj Pawar
"""
import numpy as np
import keras

from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

from scipy.integrate import odeint

#%%
def create_training_data_rnn(training_set, input_sequence_length, target_sequence_length):
    m,n = training_set.shape
    
    xtrain = np.zeros((m-input_sequence_length-target_sequence_length+1,input_sequence_length,n))
    for i in range(m-input_sequence_length-target_sequence_length+1):
        xtrain[i,:,:] = training_set[i:i+input_sequence_length,:]
    
    ytrain = np.zeros((m-input_sequence_length-target_sequence_length+1,target_sequence_length,n))
    for i in range(m-input_sequence_length-target_sequence_length+1):
        ytrain[i,:,:] = training_set[i+input_sequence_length:i+input_sequence_length+target_sequence_length,:]
        
    return xtrain, ytrain

def random_lorenz(states, input_sequence_length):
    xt = states[:input_sequence_length,:]
    yt = states[input_sequence_length:,:]
    
    return xt, yt

#%%
rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0
def f(state, t):
  x, y, z = state  # unpack the state vector
  return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # derivatives

t_init  = 0.0  # Initial time
t_final = 25.0 # Final time
dt = 0.01
t = np.arange(t_init, t_final, dt)
n = int((t_final-t_init)/dt)

state0_test = [-8.0, 7.0, 27.0]
states_test = odeint(f, state0_test, t)

nsamples = states_test.shape[0]
input_sequence_length = 5
target_sequence_length = 5
num_input_features = 3 # The dimensionality of the input at each time step. In this case a 1D signal.
num_output_features = 3 # The dimensionality of the output at each time step. In this case a 1D signal.

xtrain, ytrain = create_training_data_rnn(states_test, input_sequence_length, target_sequence_length)

#%%
decoder_input = np.zeros((xtrain.shape[0], 1, 1))

#%%
keras.backend.clear_session()

layers = [35, 35] # Number of hidden neuros in each layer of the encoder and decoder

learning_rate = 0.01
decay = 0 # Learning rate decay
optimiser = keras.optimizers.Adam(lr=learning_rate, decay=decay) # Other possible optimiser "sgd" (Stochastic Gradient Descent)

# There is no reason for the input sequence to be of same dimension as the ouput sequence.
# For instance, using 3 input signals: consumer confidence, inflation and house prices to predict the future house prices.

loss = "mse" # Other loss functions are possible, see Keras documentation.

# Regularisation isn't really needed for this application
lambda_regulariser = 0.000001 # Will not be used if regulariser is None
regulariser = None # Possible regulariser: keras.regularizers.l2(lambda_regulariser)

batch_size = 25
steps_per_epoch = 100 # batch_size * steps_per_epoch = total number of training examples
epochs = 150

#%%
# Define an input sequence.
encoder_inputs = keras.layers.Input(shape=(None, num_input_features))

# Create a list of RNN Cells, these are then concatenated into a single layer
# with the RNN layer.
encoder_cells = []
for hidden_neurons in layers:
    encoder_cells.append(keras.layers.GRUCell(hidden_neurons,
                                              kernel_regularizer=regulariser,
                                              recurrent_regularizer=regulariser,
                                              bias_regularizer=regulariser))

encoder = keras.layers.RNN(encoder_cells, return_state=True)

encoder_outputs_and_states = encoder(encoder_inputs)

# Discard encoder outputs and only keep the states.
# The outputs are of no interest to us, the encoder's
# job is to create a state describing the input sequence.
encoder_states = encoder_outputs_and_states[1:]

#%%
# The decoder input will be set to zero (see random_sine function of the utils module).
# Do not worry about the input size being 1, I will explain that in the next cell.
decoder_inputs = keras.layers.Input(shape=(None, 1))

decoder_cells = []
for hidden_neurons in layers:
    decoder_cells.append(keras.layers.GRUCell(hidden_neurons,
                                              kernel_regularizer=regulariser,
                                              recurrent_regularizer=regulariser,
                                              bias_regularizer=regulariser))

decoder = keras.layers.RNN(decoder_cells, return_sequences=True, return_state=True)

# Set the initial state of the decoder to be the ouput state of the encoder.
# This is the fundamental part of the encoder-decoder.
decoder_outputs_and_states = decoder(decoder_inputs, initial_state=encoder_states)

# Only select the output of the decoder (not the states)
decoder_outputs = decoder_outputs_and_states[0]

# Apply a dense layer with linear activation to set output to correct dimension
# and scale (tanh is default activation for GRU in Keras, our output sine function can be larger then 1)
decoder_dense = keras.layers.Dense(num_output_features,
                                   activation='linear',
                                   kernel_regularizer=regulariser,
                                   bias_regularizer=regulariser)

decoder_outputs = decoder_dense(decoder_outputs)

#%%
# Create a model using the functional API provided by Keras.
# The functional API is great, it gives an amazing amount of freedom in architecture of your NN.
# A read worth your time: https://keras.io/getting-started/functional-api-guide/ 
model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
model.compile(optimizer=optimiser, loss=loss)

#%%
filepath = "best_model.hd5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history_callback = model.fit([xtrain, decoder_input], ytrain, epochs=epochs, batch_size=batch_size, verbose=1, validation_split= 0.2,
                                    callbacks=callbacks_list)

# training and validation loss. Plot loss
loss_history = history_callback.history["loss"]
val_loss_history = history_callback.history["val_loss"]

#%%
num_steps_to_predict = nsamples - input_sequence_length

xtest = xtrain[0].reshape(1,input_sequence_length,num_input_features)
x_decoder_test = np.zeros((1,5,1))

y_test_predicted = xtest

#%%
#for k in range(num_steps_to_predict):
k = 0
while k<int(num_steps_to_predict):
    y_test_p = model.predict([xtest, x_decoder_test])
    y_test_predicted = np.hstack((y_test_predicted, y_test_p))
    xtest = np.hstack((xtest, y_test_p))
    xtest = xtest[:,input_sequence_length:,:]
    k = k + target_sequence_length

#%%
y_test_predicted = y_test_predicted.reshape(nsamples,3)

ny = 3
my = t.shape[0]
fig, axs = plt.subplots(ny, 1, figsize=(10,5))#, constrained_layout=True)

for i in range(ny):
    #axs[i].plot(ytrains[:,i], color='black', linestyle='-', label=r'$y_'+str(i+1)+'$'+' (True)', zorder=5)
    axs[i].plot(t,states_test[:,i], color='black', linestyle='-', label=r'$y_'+str(i+1)+'$'+' (True)', zorder=0)
    axs[i].plot(t,y_test_predicted[:,i], color='blue', linestyle='-.', label=r'$y_'+str(i+1)+'$'+' (GP)', zorder=5) 
    axs[i].set_xlim([t[0], t[my-1]])
    axs[i].set_ylabel('$a_'+'{'+(str(i)+'}'+'$'), fontsize = 14)

fig.tight_layout() 

fig.subplots_adjust(bottom=0.1)

line_labels = ["True", "ML"]#, "ML-Train", "ML-Test"]
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.1, ncol=6, labelspacing=0.,  prop={'size': 13} ) #bbox_to_anchor=(0.5, 0.0), borderaxespad=0.1, 

fig.savefig('gp.eps')#, bbox_inches = 'tight', pad_inches = 0.01)


#%%
encoder_predict_model = keras.models.Model(encoder_inputs,
                                           encoder_states)

decoder_states_inputs = []

# Read layers backwards to fit the format of initial_state
# For some reason, the states of the model are order backwards (state of the first layer at the end of the list)
# If instead of a GRU you were using an LSTM Cell, you would have to append two Input tensors since the LSTM has 2 states.
for hidden_neurons in layers[::-1]:
    # One state for GRU
    decoder_states_inputs.append(keras.layers.Input(shape=(hidden_neurons,)))

decoder_outputs_and_states = decoder(
    decoder_inputs, initial_state=decoder_states_inputs)

decoder_outputs = decoder_outputs_and_states[0]
decoder_states = decoder_outputs_and_states[1:]

decoder_outputs = decoder_dense(decoder_outputs)

decoder_predict_model = keras.models.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

# Let's define a small function that predicts based on the trained encoder and decoder models

def predict(x, encoder_predict_model, decoder_predict_model, num_steps_to_predict):
    """Predict time series with encoder-decoder.
    
    Uses the encoder and decoder models previously trained to predict the next
    num_steps_to_predict values of the time series.
    
    Arguments
    ---------
    x: input time series of shape (batch_size, input_sequence_length, input_dimension).
    encoder_predict_model: The Keras encoder model.
    decoder_predict_model: The Keras decoder model.
    num_steps_to_predict: The number of steps in the future to predict
    
    Returns
    -------
    y_predicted: output time series for shape (batch_size, target_sequence_length,
        ouput_dimension)
    """
    y_predicted = []

    # Encode the values as a state vector
    states = encoder_predict_model.predict(x)

    # The states must be a list
    if not isinstance(states, list):
        states = [states]

    # Generate first value of the decoder input sequence
    decoder_input = np.zeros((x.shape[0], 1, 1))


    for _ in range(num_steps_to_predict):
        outputs_and_states = decoder_predict_model.predict(
        [decoder_input] + states, batch_size=batch_size)
        output = outputs_and_states[0]
        states = outputs_and_states[1:]

        # add predicted value
        y_predicted.append(output)

    return np.concatenate(y_predicted, axis=1)

#%%
t_init  = 0.0  # Initial time
t_final = 20.0 # Final time
dt = 0.01
t = np.arange(t_init, t_final, dt)
num_steps_to_predict = int((t_final-t_init)/dt)

state0_test = [-8.0, 7.0, 27.0]
states_test = odeint(f, state0_test, t)

xtest = xtrain[0].reshape(1,lookback,num_input_features)

#%%
y_test_predicted = predict(xtest, encoder_predict_model, decoder_predict_model, num_steps_to_predict)

#%%
ny = num_input_features
my = t.shape[0]
fig, axs = plt.subplots(ny, 1, figsize=(10,5))#, constrained_layout=True)

for i in range(ny):
    #axs[i].plot(ytrains[:,i], color='black', linestyle='-', label=r'$y_'+str(i+1)+'$'+' (True)', zorder=5)
    axs[i].plot(t,states_test[:,i], color='black', linestyle='-', label=r'$y_'+str(i+1)+'$'+' (True)', zorder=0)
    axs[i].plot(t,y_test_predicted[0,:,i], color='blue', linestyle='-.', label=r'$y_'+str(i+1)+'$'+' (GP)', zorder=5) 
    axs[i].set_xlim([t[0], t[my-1]])
    axs[i].set_ylabel('$a_'+'{'+(str(i)+'}'+'$'), fontsize = 14)

fig.tight_layout() 

fig.subplots_adjust(bottom=0.1)

line_labels = ["True", "ML"]#, "ML-Train", "ML-Test"]
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.1, ncol=6, labelspacing=0.,  prop={'size': 13} ) #bbox_to_anchor=(0.5, 0.0), borderaxespad=0.1, 

fig.savefig('gp.eps')#, bbox_inches = 'tight', pad_inches = 0.01)
