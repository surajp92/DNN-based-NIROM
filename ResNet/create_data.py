#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:10:27 2019

@author: Suraj Pawar
"""
import numpy as np

def create_training_data_lstm(_training_set, _m, _n, _lookback):
    ytrain = [_training_set[i+1] for i in range(_lookback-1,_m-1)]
    ytrain = np.array(ytrain)
    xtrain = np.zeros((_m-_lookback,_lookback,_n))
    for i in range(_m-_lookback):
        a = _training_set[i]
        for j in range(1,_lookback):
            a = np.vstack((a,_training_set[i+j]))
        xtrain[i] = a
    
    return xtrain, ytrain

def create_training_data(training_set, dt, legs, slopenet):
    if (slopenet == "BDF2"):
        return create_training_data_bdf(training_set, dt, legs)
    elif (slopenet == "SEQ"):
        return create_training_data_seq(training_set, dt, legs)
    elif (slopenet == "EULER"):
        return create_training_data_euler(training_set, dt, legs)
    elif (slopenet == "RESNET"):
        return create_training_data_rn(training_set, dt, legs)
        
def create_training_data_bdf(training_set, dt, legs):
    m,n = training_set.shape
    if legs == 1:
        ytrain = [(1.5*training_set[i+1]-2.0*training_set[i]+0.5*training_set[i-1])/dt for i in range(legs, m-1)]
        ytrain = np.array(ytrain)
        xtrain = np.zeros((m-legs-1, legs*n))
        for i in range(legs,m-1):
            for k in range(legs):
                p = k*n
                xtrain[i-legs,p:p+n] = training_set[i-legs+k+1]
        return xtrain, ytrain
    else:
        ytrain = [(1.5*training_set[i+1]-2.0*training_set[i]+0.5*training_set[i-1])/dt for i in range(legs-1, m-1)]
        ytrain = np.array(ytrain)
        xtrain = np.zeros((m-legs, legs*n))
        for i in range(legs,m):
            for k in range(legs):
                p = k*n
                xtrain[i-legs,p:p+n] = training_set[i-legs+k]
        return xtrain, ytrain
    

def create_training_data_seq(training_set, dt, legs):
    m,n = training_set.shape
    ytrain = [training_set[i+1] for i in range(legs-1, m-1)]
    ytrain = np.array(ytrain)
    xtrain = np.zeros((m-legs, legs*n))
    for i in range(legs,m):
        for k in range(legs):
            p = k*n
            xtrain[i-legs,p:p+n] = training_set[i-legs+k]
    return xtrain, ytrain


def create_training_data_euler(training_set, dt, legs):
    m,n = training_set.shape
    ytrain = [(training_set[i+1]-training_set[i])/dt for i in range(legs-1, m-1)]
    ytrain = np.array(ytrain)
    xtrain = np.zeros((m-legs, legs*n))
    for i in range(legs,m):
        for k in range(legs):
            p = k*n
            xtrain[i-legs,p:p+n] = training_set[i-legs+k]
    return xtrain, ytrain



def create_training_data_rn(training_set, dt, legs):
    m,n = training_set.shape
    ytrain = [(training_set[i+1]-training_set[i]) for i in range(legs-1, m-1)]
    ytrain = np.array(ytrain)
    xtrain = np.zeros((m-legs, legs*n))
    for i in range(legs,m):
        for k in range(legs):
            p = k*n
            xtrain[i-legs,p:p+n] = training_set[i-legs+k]
    return xtrain, ytrain


