#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:10:27 2019

@author: Suraj Pawar
"""
import numpy as np


def create_training_data(training_set, m, n, dt, legs, slopenet):
    if (legs == 2) & (slopenet == "BDF"):
        return create_training_data_bdf2(training_set, m, n, dt)
    elif (legs == 3) & (slopenet == "BDF"):
        return create_training_data_bdf3(training_set, m, n, dt)
    elif (legs == 4) & (slopenet == "BDF"):
        return create_training_data_bdf4(training_set, m, n, dt)
    elif (legs == 1) & (slopenet == "LEAPFROG"):
        return create_training_data_lf1(training_set, m, n, dt)
    elif (legs == 2) & (slopenet == "LEAPFROG"):
        return create_training_data_lf2(training_set, m, n, dt)
    elif (legs == 4) & (slopenet == "LEAPFROG"):
        return create_training_data_lf4(training_set, m, n, dt)
    elif (legs == 1) & (slopenet == "SEQ"):
        return create_training_data_seq1(training_set, m, n, dt)
    elif (legs == 2) & (slopenet == "SEQ"):
        return create_training_data_seq2(training_set, m, n, dt)
    elif (legs == 1) & (slopenet == "EULER"):
        return create_training_data_e1(training_set, m, n, dt)
    elif (legs == 2) & (slopenet == "EULER"):
        return create_training_data_e2(training_set, m, n, dt)
    elif (legs == 4) & (slopenet == "EULER"):
        return create_training_data_e4(training_set, m, n, dt)
        
    
def create_training_data_bdf2(_training_set, _m, _n, _dt):
    ytrain = [(1.5*_training_set[i+1]-2.0*_training_set[i]+0.5*_training_set[i-1])/_dt for i in range(1,_m-1)]
    ytrain = np.array(ytrain)
    xtrain = [[_training_set[i-2,:], _training_set[i-1,:]] for i in range(2,_m)]
    xtrain = np.array(xtrain)
    xtrain = xtrain.reshape(_m-2,2*(_n-1))
    return xtrain, ytrain


def create_training_data_bdf3(_training_set, _m, _n, _dt):
    ytrain = [((11.0/6.0)*_training_set[i+1]-3.0*_training_set[i]+1.5*_training_set[i-1]
              - (1.0/3.0)*_training_set[i-2])/_dt for i in range(2,_m-1)]
    ytrain = np.array(ytrain)
    xtrain = [[_training_set[i-3,:], _training_set[i-2,:], _training_set[i-1,:]] for i in range(3,_m)]
    xtrain = np.array(xtrain)
    xtrain = xtrain.reshape(_m-3,3*(_n-1))
    return xtrain, ytrain


def create_training_data_bdf4(_training_set, _m, _n, _dt):
    ytrain = [((2.0/12.0)*_training_set[i+1]-4.0*_training_set[i]+3.0*_training_set[i-1]
               -(4.0/3.0)*_training_set[i-2]+(1.0/4.0)*_training_set[i-3])/_dt for i in range(3,_m-1)]
    ytrain = np.array(ytrain)
    xtrain = [[_training_set[i-4,:], _training_set[i-3,:], _training_set[i-2,:], _training_set[i-1,:]] for i in range(4,_m)]
    xtrain = np.array(xtrain)
    xtrain = xtrain.reshape(_m-4,4*(_n-1))
    return xtrain, ytrain


def create_training_data_lf1(_training_set, _m, _n, _dt):
    ytrain = [(_training_set[i+1]-_training_set[i-1])/(2.0*_dt) for i in range(1,_training_set.shape[0]-1)]
    ytrain = np.array(ytrain)
    xtrain = _training_set[1:_training_set.shape[0]-1]
    return xtrain, ytrain


def create_training_data_lf2(_training_set, _m, _n, _dt):
    ytrain = [(_training_set[i+1]-_training_set[i-1])/(2.0*_dt) for i in range(1,_m-1)]
    ytrain = np.array(ytrain)
    xtrain = [[_training_set[i-1,:], _training_set[i,:]] for i in range(1,_m-1)] # updated
    xtrain = np.array(xtrain)
    xtrain = xtrain.reshape(_m-2,2*(_n-1))
    return xtrain, ytrain


def create_training_data_lf4(_training_set, _m, _n, _dt):
    ytrain = [(_training_set[i+1]-_training_set[i-1])/(2.0*_dt) for i in range(3,_m-1)]
    ytrain = np.array(ytrain)
    xtrain = [[_training_set[i-3,:], _training_set[i-2,:], _training_set[i-1,:], _training_set[i,:]] for i in range(3,_m-1)]
    xtrain = np.array(xtrain)
    xtrain = xtrain.reshape(_m-4,4*(_n-1))
    return xtrain, ytrain


def create_training_data_seq1(_training_set, _m, _n, _dt):
    ytrain = [_training_set[i+1] for i in range(_training_set.shape[0]-1)]
    ytrain = np.array(ytrain)
    xtrain = _training_set[0:_training_set.shape[0]-1]
    return xtrain, ytrain


# updated
def create_training_data_seq2(_training_set, _m, _n, _dt):
    ytrain = [_training_set[i+2] for i in range(_m-2)]
    ytrain = np.array(ytrain)
    xtrain = [[_training_set[i-1,:], _training_set[i,:]] for i in range(1,_m-1)]
    xtrain = np.array(xtrain)
    xtrain = xtrain.reshape(_m-2,2*(_n-1))
    return xtrain, ytrain


def create_training_data_e1(_training_set, _m, _n, _dt):
    ytrain = [(_training_set[i+1]-_training_set[i])/_dt for i in range(_training_set.shape[0]-1)]
    ytrain = np.array(ytrain)
    xtrain = _training_set[0:_training_set.shape[0]-1]
    return xtrain, ytrain


def create_training_data_e2(_training_set, _m, _n, _dt):
    ytrain = [(_training_set[i+1]-_training_set[i])/_dt for i in range(1,_m-1)]
    ytrain = np.array(ytrain)
    xtrain = [[_training_set[i-1,:], _training_set[i,:]] for i in range(1,_m-1)]
    xtrain = np.array(xtrain)
    xtrain = xtrain.reshape(_m-2,2*(_n-1))
    return xtrain, ytrain


def create_training_data_e4(_training_set, _m, _n, _dt):
    ytrain = [(_training_set[i+1]-_training_set[i])/_dt for i in range(3,_m-1)]
    ytrain = np.array(ytrain)
    xtrain = [[_training_set[i-3,:], _training_set[i-2,:], _training_set[i-1,:], _training_set[i,:]] for i in range(3,_m-1)]
    xtrain = np.array(xtrain)
    xtrain = xtrain.reshape(_m-4,4*(_n-1))
    return xtrain, ytrain
