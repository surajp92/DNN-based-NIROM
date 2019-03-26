#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:10:27 2019

@author: Suraj Pawar
"""
import numpy as np

def create_training_data(training_set, m, n, dt, legs, slopnet):
    if (legs == 2) & (slopnet == "BDF"):
        xtrain, ytrain = create_training_data_bdf2(training_set, m, n, dt)
        return xtrain, ytrain
    elif (legs == 3) & (slopnet == "BDF"):
        xtrain, ytrain = create_training_data_bdf3(training_set, m, n, dt)
        return xtrain, ytrain
    elif (legs == 4) & (slopnet == "BDF"):
        xtrain, ytrain = create_training_data_bdf4(training_set, m, n, dt)
        return xtrain, ytrain
    elif (legs == 1) & (slopnet == "LEAPFROG"):
        xtrain, ytrain = create_training_data_lf1(training_set, m, n, dt)
        return xtrain, ytrain
    elif (legs == 2) & (slopnet == "LEAPFROG"):
        xtrain, ytrain = create_training_data_lf2(training_set, m, n, dt)
        return xtrain, ytrain
    elif (legs == 4) & (slopnet == "LEAPFROG"):
        xtrain, ytrain = create_training_data_lf4(training_set, m, n, dt)
        return xtrain, ytrain
    elif (legs == 1) & (slopnet == "SEQ"):
        xtrain, ytrain = create_training_data_seq1(training_set, m, n, dt)
        return xtrain, ytrain
        
    
def create_training_data_bdf2(training_set, m, n, dt):
    ytrain = [(1.5*training_set[i+1]-2.0*training_set[i]+0.5*training_set[i-1])/(dt) for i in range(1,m-1)]
    ytrain = np.array(ytrain)
    xtrain = [[training_set[i-2,:], training_set[i-1,:]] for i in range(2,m)]
    xtrain = np.array(xtrain)
    xtrain = xtrain.reshape(m-2,20)
    return xtrain, ytrain

def create_training_data_bdf3(training_set, m, n, dt):
    ytrain = [((11.0/6.0)*training_set[i+1]-3.0*training_set[i]+1.5*training_set[i-1]
          -(1.0/3.0)*training_set[i-2])/(dt) for i in range(2,m-1)]
    ytrain = np.array(ytrain)
    xtrain = [[training_set[i-3,:], training_set[i-2,:], training_set[i-1,:]] for i in range(3,m)]
    xtrain = np.array(xtrain)
    xtrain = xtrain.reshape(m-3,3*(n-1))
    return xtrain, ytrain

def create_training_data_bdf4(training_set, m, n, dt):
    ytrain = [((2.0/12.0)*training_set[i+1]-4.0*training_set[i]+3.0*training_set[i-1]
          -(4.0/3.0)*training_set[i-2]+(1.0/4.0)*training_set[i-3])/(dt) for i in range(3,m-1)]
    ytrain = np.array(ytrain)
    xtrain = [[training_set[i-4,:], training_set[i-3,:], training_set[i-2,:], training_set[i-1,:]] for i in range(4,m)]
    xtrain = np.array(xtrain)
    xtrain = xtrain.reshape(m-4,4*(n-1))
    return xtrain, ytrain

def create_training_data_lf1(training_set, m, n, dt):
    ytrain = [(training_set[i+1]-training_set[i-1])/(2.0*dt) for i in range(1,training_set.shape[0]-1)]
    ytrain = np.array(ytrain)
    xtrain = training_set[1:training_set.shape[0]-1]
    return xtrain, ytrain

def create_training_data_lf2(training_set, m, n, dt):
    ytrain = [(training_set[i+1]-training_set[i-1])/(2.0*dt) for i in range(1,m-1)]
    ytrain = np.array(ytrain)
    xtrain = [[training_set[i-2,:], training_set[i-1,:]] for i in range(2,m)]
    xtrain = np.array(xtrain)
    xtrain = xtrain.reshape(m-2,20)
    return xtrain, ytrain

def create_training_data_lf4(training_set, m, n, dt):
    ytrain = [(training_set[i+1]-training_set[i-1])/(2.0*dt) for i in range(3,m-1)]
    ytrain = np.array(ytrain)
    xtrain = [[training_set[i-3,:], training_set[i-2,:], training_set[i-1,:], training_set[i,:]] for i in range(3,m-1)]
    xtrain = np.array(xtrain)
    xtrain = xtrain.reshape(m-4,40)
    return xtrain, ytrain

def create_training_data_seq1(training_set, m, n, dt):
    ytrain = [training_set[i+1] for i in range(training_set.shape[0]-1)]
    ytrain = np.array(ytrain)
    xtrain = training_set[0:training_set.shape[0]-1]
    return xtrain, ytrain