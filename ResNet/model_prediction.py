#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:38:42 2019

@author: Suraj Pawar
"""
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model, Model
from keras import backend as K

#--------------------------------------------------------------------------------------------------------------#
def coeff_determination(y_true, y_pred):
        SS_res =  K.sum(K.square( y_true-y_pred ))
        SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
        return ( 1 - SS_res/(SS_tot + K.epsilon()) )
    
#--------------------------------------------------------------------------------------------------------------#
def customloss(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

def model_predict_lstm(testing_set, _m, n, _lookback):
    
    ytest = np.zeros((1,_lookback,n))
    ytest_ml = np.zeros((_m,n))
    # create input at t= 0 for the model testing
    for i in range(_lookback):
        ytest[0,i,:] = testing_set[i]
        ytest_ml[i] = testing_set[i]
           
    for i in range(_lookback,_m):
        slope_ml = model.predict(ytest) # slope from LSTM/ ML model
        ytest_ml[i] = slope_ml
        e = ytest
        for i in range(_lookback-1):
            e[0,i,:] = e[0,i+1,:]
        e[0,_lookback-1,:] = slope_ml
        ytest = e
    return ytest_ml
    

def model_predict(testing_set, dt, legs, slopenet, sc_input, sc_output):
    if  (slopenet == "BDF2"):
        return model_predict_bdf(testing_set, dt, sc_input, sc_output, legs)
    elif  (slopenet == "SEQ"):
        return model_predict_seq(testing_set, dt, sc_input, sc_output, legs)
    elif  (slopenet == "EULER"):
        return model_predict_euler(testing_set, dt, sc_input, sc_output, legs)
    elif  (slopenet == "RESNET"):
        return model_predict_rn(testing_set, dt, sc_input, sc_output, legs)
        #return model_predict_test(testing_set, dt, sc_input, sc_output)


def model_predict_bdf(testing_set, dt, sc_input, sc_output, legs):
    print("bdf")
    m,n = testing_set.shape
    custom_model = load_model('best_model.hd5')#,custom_objects={'coeff_determination': coeff_determination})
    ytest = np.zeros((m,n*legs))
    if legs ==1:
        ytest[0,:] = testing_set[1,:]
        
        yt_ml = np.zeros((m,n))
        for k in range(legs+1):
            yt_ml[k] = testing_set[k,:] 
        
        xt_sc = np.zeros((1,legs*n))
        yt_sc = np.zeros((1,n))
        
        for i in range(legs+1,m):
            xt = ytest[i-legs-1]
            xt = xt.reshape(1,legs*n)
            xt_sc = sc_input.transform(xt)
            yt = custom_model.predict(xt_sc) 
            yt_sc = sc_output.inverse_transform(yt)
            yt_ml[i] = (4.0/3.0)*yt_ml[i-1]- (1.0/3.0)*yt_ml[i-2] + (2.0/3.0)*dt*yt_sc
            q = (legs-1)*n
            ytest[i-legs,:q] = ytest[i-legs-1,n:]
            ytest[i-legs,q:] = yt_ml[i]
    else:
        for k in range(legs):
            p = k*n
            ytest[0,p:p+n] = testing_set[k,:]
        
        yt_ml = np.zeros((m,n))
        for k in range(legs):
            yt_ml[k] = testing_set[k,:] 
        
        xt_sc = np.zeros((1,legs*n))
        yt_sc = np.zeros((1,n))
        
        for i in range(legs,m):
            xt = ytest[i-legs]
            xt = xt.reshape(1,legs*n)
            xt_sc = sc_input.transform(xt)
            yt = custom_model.predict(xt_sc) 
            yt_sc = sc_output.inverse_transform(yt)
            yt_ml[i] = (4.0/3.0)*yt_ml[i-1]- (1.0/3.0)*yt_ml[i-2] + (2.0/3.0)*dt*yt_sc
            q = (legs-1)*n
            ytest[i-legs+1,:q] = ytest[i-legs,n:]
            ytest[i-legs+1,q:] = yt_ml[i]
        
    return yt_ml
    

def model_predict_seq(testing_set, dt, sc_input, sc_output, legs):
    print("seq")
    m,n = testing_set.shape
    custom_model = load_model('best_model.hd5')#,custom_objects={'coeff_determination': coeff_determination})
    ytest = np.zeros((m,n*legs))
    for k in range(legs):
        p = k*n
        ytest[0,p:p+n] = testing_set[k,:]
    
    yt_ml = np.zeros((m,n))
    for k in range(legs):
        yt_ml[k] = testing_set[k,:] 
    
    xt_sc = np.zeros((1,legs*n))
    yt_sc = np.zeros((1,n))
    
    for i in range(legs,m):
        xt = ytest[i-legs]
        xt = xt.reshape(1,legs*n)
        xt_sc = sc_input.transform(xt)
        yt = custom_model.predict(xt_sc) 
        yt_sc = sc_output.inverse_transform(yt)
        yt_ml[i] =  yt_sc
        q = (legs-1)*n
        ytest[i-legs+1,:q] = ytest[i-legs,n:]
        ytest[i-legs+1,q:] = yt_ml[i]
    
    return yt_ml


def model_predict_euler(testing_set, dt, sc_input, sc_output, legs):
    print("euler")
    m,n = testing_set.shape
    custom_model = load_model('best_model.hd5')#,custom_objects={'coeff_determination': coeff_determination})
    ytest = np.zeros((m,n*legs))
    for k in range(legs):
        p = k*n
        ytest[0,p:p+n] = testing_set[k,:]
    
    yt_ml = np.zeros((m,n))
    for k in range(legs):
        yt_ml[k] = testing_set[k,:] 
    
    xt_sc = np.zeros((1,legs*n))
    yt_sc = np.zeros((1,n))
    
    for i in range(legs,m):
        xt = ytest[i-legs]
        xt = xt.reshape(1,legs*n)
        xt_sc = sc_input.transform(xt)
        yt = custom_model.predict(xt_sc) 
        yt_sc = sc_output.inverse_transform(yt)
        yt_ml[i] =  yt_ml[i-1] + yt_sc*dt
        q = (legs-1)*n
        ytest[i-legs+1,:q] = ytest[i-legs,n:]
        ytest[i-legs+1,q:] = yt_ml[i]
    
    return yt_ml



def model_predict_rn(testing_set, dt, sc_input, sc_output, legs):
    print("resnet")
    m,n = testing_set.shape
    custom_model = load_model('best_model.hd5')#,custom_objects={'coeff_determination': coeff_determination})
    ytest = np.zeros((m,n*legs))
    for k in range(legs):
        p = k*n
        ytest[0,p:p+n] = testing_set[k,:]
    
    yt_ml = np.zeros((m,n))
    for k in range(legs):
        yt_ml[k] = testing_set[k,:] 
    
    xt_sc = np.zeros((1,legs*n))
    yt_sc = np.zeros((1,n))
    
    for i in range(legs,m):
        xt = ytest[i-legs]
        xt = xt.reshape(1,legs*n)
        xt_sc = sc_input.transform(xt)
        yt = custom_model.predict(xt_sc) 
        yt_sc = sc_output.inverse_transform(yt)
        yt_ml[i] =  yt_ml[i-1] + yt_sc
        q = (legs-1)*n
        ytest[i-legs+1,:q] = ytest[i-legs,n:]
        ytest[i-legs+1,q:] = yt_ml[i]
    
    return yt_ml
    

def model_predict_test(testing_set, dt, sc_input, sc_output):
    print("rn4")
    custom_model = load_model('best_model.hd5')#,custom_objects={'coeff_determination': coeff_determination})
    m,n = testing_set.shape
    n = n+1
    
    ytest = [testing_set[0], testing_set[1], testing_set[2], testing_set[3]]
    ytest = np.array(ytest)
    ytest = ytest.reshape(1,4*(n-1))

    ytest_ml = [testing_set[0]]
    ytest_ml = np.array(ytest_ml)
    ytest_ml = np.vstack((ytest_ml, testing_set[1]))
    ytest_ml = np.vstack((ytest_ml, testing_set[2]))
    ytest_ml = np.vstack((ytest_ml, testing_set[3]))

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

    return ytest_ml