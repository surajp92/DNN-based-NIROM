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
    

def model_predict(testing_set, m, n, dt, legs, slopenet, sigma, sc_input, sc_output):
    if (legs == 1) & (slopenet == "BDF2"):
        return model_predict_bdf21(testing_set, m, n, dt, sc_input, sc_output)
    elif (legs == 2) & (slopenet == "BDF2"):
        return model_predict_bdf22(testing_set, m, n, dt, sc_input, sc_output)
    elif (legs == 4) & (slopenet == "BDF2"):
        return model_predict_bdf24(testing_set, m, n, dt, sc_input, sc_output)
    elif (legs == 1) & (slopenet == "BDF3"):
        return model_predict_bdf31(testing_set, m, n, dt, sc_input, sc_output)
    elif (legs == 2) & (slopenet == "BDF3"):
        return model_predict_bdf32(testing_set, m, n, dt, sc_input, sc_output)
    elif (legs == 3) & (slopenet == "BDF3"):
        return model_predict_bdf33(testing_set, m, n, dt, sc_input, sc_output)
    elif (legs == 4) & (slopenet == "BDF3"):
        return model_predict_bdf34(testing_set, m, n, dt, sc_input, sc_output)
    elif (legs == 1) & (slopenet == "BDF4"):
        return model_predict_bdf41(testing_set, m, n, dt, sc_input, sc_output)
    elif (legs == 2) & (slopenet == "BDF4"):
        return model_predict_bdf42(testing_set, m, n, dt, sc_input, sc_output)
    elif (legs == 4) & (slopenet == "BDF4"):
        return model_predict_bdf44(testing_set, m, n, dt, sc_input, sc_output)
    elif (legs == 1) & (slopenet == "LEAPFROG"):
        return model_predict_lf1(testing_set, m, n, dt, sc_input, sc_output)
    elif (legs == 2) & (slopenet == "LEAPFROG"):
        return model_predict_lf2(testing_set, m, n, dt, sc_input, sc_output)
    elif (legs == 4) & (slopenet == "LEAPFROG"):
        return model_predict_lf4(testing_set, m, n, dt, sc_input, sc_output)
    elif (legs == 1) & (slopenet == "LEAPFROG-FILTER"):
        return model_predict_lff1(testing_set, m, n, dt, sigma, sc_input, sc_output)
    elif (legs == 2) & (slopenet == "LEAPFROG-FILTER"):
        return model_predict_lff2(testing_set, m, n, dt, sigma, sc_input, sc_output)
    elif (legs == 4) & (slopenet == "LEAPFROG-FILTER"):
        return model_predict_lff4(testing_set, m, n, dt, sigma, sc_input, sc_output)
    elif (legs == 1) & (slopenet == "SEQ"):
        return model_predict_seq1(testing_set, m, n, dt, sc_input, sc_output)
    elif (legs == 2) & (slopenet == "SEQ"):
         return model_predict_seq2(testing_set, m, n, dt, sc_input, sc_output)
    elif (legs == 4) & (slopenet == "SEQ"):
         return model_predict_seq4(testing_set, m, n, dt, sc_input, sc_output)
    elif (legs == 1) & (slopenet == "EULER"):
        return model_predict_e1(testing_set, m, n, dt, sc_input, sc_output)
    elif (legs == 2) & (slopenet == "EULER"):
        return model_predict_e2(testing_set, m, n, dt, sc_input, sc_output)
    elif (legs == 4) & (slopenet == "EULER"):
        return model_predict_e4(testing_set, m, n, dt, sc_input, sc_output)
    elif (legs == 1) & (slopenet == "RESNET"):
        return model_predict_rn1(testing_set, m, n, dt, sc_input, sc_output)
    elif (legs == 2) & (slopenet == "RESNET"):
        return model_predict_rn2(testing_set, m, n, dt, sc_input, sc_output)
    elif (legs == 4) & (slopenet == "RESNET"):
        return model_predict_rn4(testing_set, m, n, dt, sc_input, sc_output)

def model_predict_bdf21(testing_set, m, n, dt, sc_input, sc_output):
    print("bdf21")
    custom_model = load_model('best_model.hd5',custom_objects={'coeff_determination': coeff_determination})
    
    # create input at t= 0 for the model testing
    ytest = [testing_set[1]]
    ytest = np.array(ytest)
    ytest = ytest.reshape(1,1*(n-1))
    
    ytest_ml = [testing_set[0]]
    ytest_ml = np.array(ytest_ml)
    ytest_ml = np.vstack((ytest_ml, testing_set[1]))
      
    for i in range(2,m):
        ytest_sc = sc_input.transform(ytest)
        slope_ml = custom_model.predict(ytest_sc) # slope from LSTM/ ML model
        slope_ml_sc = sc_output.inverse_transform(slope_ml)
        a = (4.0/3.0)*ytest_ml[i-1]- (1.0/3.0)*ytest_ml[i-2] + (2.0/3.0)*dt*slope_ml_sc[0] # y1 at next time step
        ytest_ml = np.vstack((ytest_ml, a))
        e = a.reshape(1,n-1)
        ytemp = ytest[0]
        ytemp = ytemp.reshape(1,1*(n-1))
        ee = np.concatenate((ytemp,e), axis = 1)
        ee = ee[0,n-1:]
        ytest = ee.reshape(1,1*(n-1)) #np.vstack((ytest, ee)) # add [y1, y2, y3] at (n+1) to input test for next slope prediction
    
    return ytest_ml

def model_predict_bdf22(testing_set, m, n, dt, sc_input, sc_output):
    print("bdf22")
    custom_model = load_model('best_model.hd5',custom_objects={'coeff_determination': coeff_determination})
    
    # create input at t= 0 for the model testing
    ytest = [testing_set[0], testing_set[1]]
    ytest = np.array(ytest)
    ytest = ytest.reshape(1,2*(n-1))
    
    ytest_ml = [testing_set[0]]
    ytest_ml = np.array(ytest_ml)
    ytest_ml = np.vstack((ytest_ml, testing_set[1]))
      
    for i in range(2,m):
        ytest_sc = sc_input.transform(ytest)
        slope_ml = custom_model.predict(ytest_sc) # slope from LSTM/ ML model
        slope_ml_sc = sc_output.inverse_transform(slope_ml)
        a = (4.0/3.0)*ytest_ml[i-1]- (1.0/3.0)*ytest_ml[i-2] + (2.0/3.0)*dt*slope_ml_sc[0] # y1 at next time step
        ytest_ml = np.vstack((ytest_ml, a))
        e = a.reshape(1,n-1)
        ytemp = ytest[0]
        ytemp = ytemp.reshape(1,2*(n-1))
        ee = np.concatenate((ytemp,e), axis = 1)
        ee = ee[0,n-1:]
        ytest = ee.reshape(1,2*(n-1)) #np.vstack((ytest, ee)) # add [y1, y2, y3] at (n+1) to input test for next slope prediction
    
    return ytest_ml

def model_predict_bdf24(testing_set, m, n, dt, sc_input, sc_output):
    print("bdf24")
    custom_model = load_model('best_model.hd5',custom_objects={'coeff_determination': coeff_determination})
    
    # create input at t= 0 for the model testing
    ytest = [testing_set[0], testing_set[1], testing_set[2], testing_set[3]]
    ytest = np.array(ytest)
    ytest = ytest.reshape(1,4*(n-1))
    
    ytest_ml = [testing_set[0]]
    ytest_ml = np.array(ytest_ml)
    ytest_ml = np.vstack((ytest_ml, testing_set[1]))
    ytest_ml = np.vstack((ytest_ml, testing_set[2]))
    ytest_ml = np.vstack((ytest_ml, testing_set[3]))
      
    for i in range(4,m):
        ytest_sc = sc_input.transform(ytest)
        slope_ml = custom_model.predict(ytest_sc) # slope from LSTM/ ML model
        slope_ml_sc = sc_output.inverse_transform(slope_ml)
        a = (4.0/3.0)*ytest_ml[i-1]- (1.0/3.0)*ytest_ml[i-2] + (2.0/3.0)*dt*slope_ml_sc[0] # y1 at next time step
        ytest_ml = np.vstack((ytest_ml, a))
        e = a.reshape(1,n-1)
        ytemp = ytest[0]
        ytemp = ytemp.reshape(1,4*(n-1))
        ee = np.concatenate((ytemp,e), axis = 1)
        ee = ee[0,n-1:]
        ytest = ee.reshape(1,4*(n-1)) 
    
    return ytest_ml

def model_predict_bdf31(testing_set, m, n, dt, sc_input, sc_output):
    print("bdf31")
    custom_model = load_model('best_model.hd5',custom_objects={'coeff_determination': coeff_determination})
    
    # create input at t= 0 for the model testing
    ytest = [testing_set[2]]
    ytest = np.array(ytest)
    ytest = ytest.reshape(1,1*(n-1))
    
    ytest_ml = [testing_set[0]]
    ytest_ml = np.array(ytest_ml)
    ytest_ml = np.vstack((ytest_ml, testing_set[1]))
    ytest_ml = np.vstack((ytest_ml, testing_set[2]))
    
    for i in range(3,m):
        ytest_sc = sc_input.transform(ytest)
        slope_ml = custom_model.predict(ytest_sc) # slope from LSTM/ ML model
        slope_ml_sc = sc_output.inverse_transform(slope_ml)
        a = (18.0/11.0)*ytest_ml[i-1]- (9.0/11.0)*ytest_ml[i-2]+(2.0/11.0)* ytest_ml[i-3]+ (6.0/11.0)*dt*slope_ml_sc[0] 
        ytest_ml = np.vstack((ytest_ml, a))
        e = a.reshape(1,n-1)
        ytemp = ytest[0]
        ytemp = ytemp.reshape(1,1*(n-1))
        ee = np.concatenate((ytemp,e), axis = 1)
        ee = ee[0,n-1:]
        ytest = ee.reshape(1,1*(n-1)) 
        
    return ytest_ml

def model_predict_bdf32(testing_set, m, n, dt, sc_input, sc_output):
    print("bdf32")
    custom_model = load_model('best_model.hd5',custom_objects={'coeff_determination': coeff_determination})
    
    # create input at t= 0 for the model testing
    ytest = [ testing_set[1], testing_set[2]]
    ytest = np.array(ytest)
    ytest = ytest.reshape(1,2*(n-1))
    
    ytest_ml = [testing_set[0]]
    ytest_ml = np.array(ytest_ml)
    ytest_ml = np.vstack((ytest_ml, testing_set[1]))
    ytest_ml = np.vstack((ytest_ml, testing_set[2]))
    
    for i in range(3,m):
        ytest_sc = sc_input.transform(ytest)
        slope_ml = custom_model.predict(ytest_sc) # slope from LSTM/ ML model
        slope_ml_sc = sc_output.inverse_transform(slope_ml)
        a = (18.0/11.0)*ytest_ml[i-1]- (9.0/11.0)*ytest_ml[i-2]+(2.0/11.0)* ytest_ml[i-3]+ (6.0/11.0)*dt*slope_ml_sc[0] # y1 at next time step
        ytest_ml = np.vstack((ytest_ml, a))
        e = a.reshape(1,n-1)
        ytemp = ytest[0]
        ytemp = ytemp.reshape(1,2*(n-1))
        ee = np.concatenate((ytemp,e), axis = 1)
        ee = ee[0,n-1:]
        ytest = ee.reshape(1,2*(n-1)) 
        
    return ytest_ml

def model_predict_bdf33(testing_set, m, n, dt, sc_input, sc_output):
    print("bdf33")
    custom_model = load_model('best_model.hd5',custom_objects={'coeff_determination': coeff_determination})
    
    # create input at t= 0 for the model testing
    ytest = [testing_set[0], testing_set[1], testing_set[2]]
    ytest = np.array(ytest)
    ytest = ytest.reshape(1,3*(n-1))
    
    ytest_ml = [testing_set[0]]
    ytest_ml = np.array(ytest_ml)
    ytest_ml = np.vstack((ytest_ml, testing_set[1]))
    ytest_ml = np.vstack((ytest_ml, testing_set[2]))
    
    for i in range(3,m):
        ytest_sc = sc_input.transform(ytest)
        slope_ml = custom_model.predict(ytest_sc) # slope from LSTM/ ML model
        slope_ml_sc = sc_output.inverse_transform(slope_ml)
        a = (18.0/11.0)*ytest_ml[i-1]- (9.0/11.0)*ytest_ml[i-2]+(2.0/11.0)* ytest_ml[i-3]+ (6.0/11.0)*dt*slope_ml_sc[0] # y1 at next time step
        ytest_ml = np.vstack((ytest_ml, a))
        e = a.reshape(1,n-1)
        ytemp = ytest[0]
        ytemp = ytemp.reshape(1,3*(n-1))
        ee = np.concatenate((ytemp,e), axis = 1)
        ee = ee[0,n-1:]
        ytest = ee.reshape(1,3*(n-1)) 
        
    return ytest_ml

def model_predict_bdf34(testing_set, m, n, dt, sc_input, sc_output):
    print("bdf34")
    custom_model = load_model('best_model.hd5',custom_objects={'coeff_determination': coeff_determination})
    
    # create input at t= 0 for the model testing
    ytest = [testing_set[0], testing_set[1], testing_set[2], testing_set[3]]
    ytest = np.array(ytest)
    ytest = ytest.reshape(1,4*(n-1))
    
    ytest_ml = [testing_set[0]]
    ytest_ml = np.array(ytest_ml)
    ytest_ml = np.vstack((ytest_ml, testing_set[1]))
    ytest_ml = np.vstack((ytest_ml, testing_set[2]))
    ytest_ml = np.vstack((ytest_ml, testing_set[3]))
    
    for i in range(4,m):
        ytest_sc = sc_input.transform(ytest)
        slope_ml = custom_model.predict(ytest_sc) # slope from LSTM/ ML model
        slope_ml_sc = sc_output.inverse_transform(slope_ml)
        a = (18.0/11.0)*ytest_ml[i-1]- (9.0/11.0)*ytest_ml[i-2]+(2.0/11.0)* ytest_ml[i-3]+ (6.0/11.0)*dt*slope_ml_sc[0] # y1 at next time step
        ytest_ml = np.vstack((ytest_ml, a))
        e = a.reshape(1,n-1)
        ytemp = ytest[0]
        ytemp = ytemp.reshape(1,4*(n-1))
        ee = np.concatenate((ytemp,e), axis = 1)
        ee = ee[0,n-1:]
        ytest = ee.reshape(1,4*(n-1)) 
        
    return ytest_ml

def model_predict_bdf41(testing_set, m, n, dt, sc_input, sc_output):
    print("bdf41")
    custom_model = load_model('best_model.hd5',custom_objects={'coeff_determination': coeff_determination})
    
    # create input at t= 0 for the model testing
    ytest = [testing_set[3]]
    ytest = np.array(ytest)
    ytest = ytest.reshape(1,1*(n-1))
    
    ytest_ml = [testing_set[0]]
    ytest_ml = np.array(ytest_ml)
    ytest_ml = np.vstack((ytest_ml, testing_set[1]))
    ytest_ml = np.vstack((ytest_ml, testing_set[2]))
    ytest_ml = np.vstack((ytest_ml, testing_set[3]))
    
    for i in range(4,m):
        ytest_sc = sc_input.transform(ytest)
        slope_ml = custom_model.predict(ytest_sc) # slope from LSTM/ ML model
        slope_ml_sc = sc_output.inverse_transform(slope_ml)
        a = (48.0/25.0)*ytest_ml[i-1]- (36.0/25.0)*ytest_ml[i-2]+(16.0/25.0)*ytest_ml[i-3]-(3.0/25.0)*ytest_ml[i-4]+(12.0/25.0)*dt*slope_ml_sc[0] # y1 at next time step
        ytest_ml = np.vstack((ytest_ml, a))
        e = a.reshape(1,n-1)
        ytemp = ytest[0]
        ytemp = ytemp.reshape(1,1*(n-1))
        ee = np.concatenate((ytemp,e), axis = 1)
        ee = ee[0,n-1:]
        ytest = ee.reshape(1,1*(n-1))
    
    return ytest_ml

def model_predict_bdf42(testing_set, m, n, dt, sc_input, sc_output):
    print("bdf42")
    custom_model = load_model('best_model.hd5',custom_objects={'coeff_determination': coeff_determination})
    
    # create input at t= 0 for the model testing
    ytest = [testing_set[2],testing_set[3]]
    ytest = np.array(ytest)
    ytest = ytest.reshape(1,2*(n-1))
    
    ytest_ml = [testing_set[0]]
    ytest_ml = np.array(ytest_ml)
    ytest_ml = np.vstack((ytest_ml, testing_set[1]))
    ytest_ml = np.vstack((ytest_ml, testing_set[2]))
    ytest_ml = np.vstack((ytest_ml, testing_set[3]))
    
    for i in range(4,m):
        ytest_sc = sc_input.transform(ytest)
        slope_ml = custom_model.predict(ytest_sc) # slope from LSTM/ ML model
        slope_ml_sc = sc_output.inverse_transform(slope_ml)
        a = (48.0/25.0)*ytest_ml[i-1]- (36.0/25.0)*ytest_ml[i-2]+(16.0/25.0)*ytest_ml[i-3]-(3.0/25.0)*ytest_ml[i-4]+(12.0/25.0)*dt*slope_ml_sc[0] # y1 at next time step
        ytest_ml = np.vstack((ytest_ml, a))
        e = a.reshape(1,n-1)
        ytemp = ytest[0]
        ytemp = ytemp.reshape(1,2*(n-1))
        ee = np.concatenate((ytemp,e), axis = 1)
        ee = ee[0,n-1:]
        ytest = ee.reshape(1,2*(n-1))
    
    return ytest_ml

def model_predict_bdf44(testing_set, m, n, dt, sc_input, sc_output):
    print("bdf44")
    custom_model = load_model('best_model.hd5',custom_objects={'coeff_determination': coeff_determination})
    
    # create input at t= 0 for the model testing
    ytest = [testing_set[0], testing_set[1], testing_set[2], testing_set[3]]
    ytest = np.array(ytest)
    ytest = ytest.reshape(1,4*(n-1))
    
    ytest_ml = [testing_set[0]]
    ytest_ml = np.array(ytest_ml)
    ytest_ml = np.vstack((ytest_ml, testing_set[1]))
    ytest_ml = np.vstack((ytest_ml, testing_set[2]))
    ytest_ml = np.vstack((ytest_ml, testing_set[3]))
    
    for i in range(4,m):
        ytest_sc = sc_input.transform(ytest)
        slope_ml = custom_model.predict(ytest_sc) # slope from LSTM/ ML model
        slope_ml_sc = sc_output.inverse_transform(slope_ml)
        a = (48.0/25.0)*ytest_ml[i-1]- (36.0/25.0)*ytest_ml[i-2]+(16.0/25.0)*ytest_ml[i-3]-(3.0/25.0)*ytest_ml[i-4]+(12.0/25.0)*dt*slope_ml_sc[0] # y1 at next time step
        ytest_ml = np.vstack((ytest_ml, a))
        e = a.reshape(1,n-1)
        ytemp = ytest[0]
        ytemp = ytemp.reshape(1,4*(n-1))
        ee = np.concatenate((ytemp,e), axis = 1)
        ee = ee[0,n-1:]
        ytest = ee.reshape(1,4*(n-1))
    
    return ytest_ml


def model_predict_lf1(testing_set, m, n, dt, sc_input, sc_output):
    print("lf1")
    custom_model = load_model('best_model.hd5',custom_objects={'coeff_determination': coeff_determination})
    
    # create input at t= 0 for the model testing
    ytest = [testing_set[0]]
    ytest = np.array(ytest)
    
    ytest_ml = [testing_set[0]]
    ytest_ml = np.array(ytest_ml)
    ytest_ml = np.vstack((ytest_ml, testing_set[1]))
    
    for i in range(1,m):
        ytest_sc = sc_input.transform(ytest)
        slope_ml = custom_model.predict(ytest_sc) # slope from LSTM/ ML model
        slope_ml_sc = sc_output.inverse_transform(slope_ml)
        a = ytest_ml[i-1] + 2.0*dt*slope_ml_sc[0] # y1 at next time step
        ytest_ml = np.vstack((ytest_ml, a))
        ytest = a.reshape(1,(n-1))
        
    return ytest_ml
    
def model_predict_lff1(testing_set, m, n, dt, sigma, sc_input, sc_output):
    print("lff1")
    custom_model = load_model('best_model.hd5',custom_objects={'coeff_determination': coeff_determination})
    
    ytest = [testing_set[1]]
    ytest = np.array(ytest)
    # ytest = ytest.reshape(1,3)
    
    ytest_ml = np.zeros((m,n-1))
    ytest_ml[0] = testing_set[0]
    ytest_ml[1] = testing_set[1]
    yf_ml = np.zeros((_m,n-1))
    yf_ml[0] = testing_set[0]
    
    for i in range(1,m):
        ytest_sc = sc_input.transform(ytest)
        slope_ml = custom_model.predict(ytest_sc) # slope from LSTM/ ML model
        slope_ml_sc = sc_output.inverse_transform(slope_ml)
        a = yf_ml[i-1] + 2.0*dt*slope_ml_sc
        ytest_ml[i+1] = a
        yf_ml[i] = ytest_ml[i] + (sigma / 2) * (yf_ml[i - 1] - 2.0 * ytest_ml[i] + ytest_ml[i+1] )
        ytest = a.reshape(1,n-1)
        
    return ytest_ml


def model_predict_lf2(testing_set, m, n, dt, sc_input, sc_output):
    print("lf2")
    custom_model = load_model('best_model.hd5',custom_objects={'coeff_determination': coeff_determination})
    
    ytest = [testing_set[0], testing_set[1]]
    ytest = np.array(ytest)
    ytest = ytest.reshape(1,2*(n-1))
    
    ytest_ml = [testing_set[0]]
    ytest_ml = np.array(ytest_ml)
    ytest_ml = np.vstack((ytest_ml, testing_set[1]))
    
    for i in range(1,m):
        ytest_sc = sc_input.transform(ytest)
        slope_ml = custom_model.predict(ytest_sc) # slope from LSTM/ ML model
        slope_ml_sc = sc_output.inverse_transform(slope_ml)
        a = ytest_ml[i-1] + 2.0*dt*slope_ml_sc[0] # y1 at next time step
        ytest_ml = np.vstack((ytest_ml, a))
        e = a.reshape(1,n-1)
        ytemp = ytest[0]
        ytemp = ytemp.reshape(1,2*(n-1))
        ee = np.concatenate((ytemp,e), axis = 1)
        ee = ee[0,n-1:]
        ytest = ee.reshape(1,2*(n-1))
    
    return ytest_ml


def model_predict_lff2(testing_set, m, n, dt, sigma, sc_input, sc_output):
    print("lff2")
    custom_model = load_model('best_model.hd5',custom_objects={'coeff_determination': coeff_determination})
    
    ytest = [testing_set[0], testing_set[1]]
    ytest = np.array(ytest)
    ytest = ytest.reshape(1,2*(n-1))
    
    ytest_ml = [testing_set[0]]
    ytest_ml = np.array(ytest_ml)
    ytest_ml = np.vstack((ytest_ml, testing_set[1]))
    yf_ml = [testing_set[0]]
    yf_ml = np.array(yf_ml)
    
    for i in range(1,m):
        ytest_sc = sc_input.transform(ytest)
        slope_ml = custom_model.predict(ytest_sc) # slope from LSTM/ ML model
        slope_ml_sc = sc_output.inverse_transform(slope_ml)
        a = yf_ml[i-1] + 2.0*dt*slope_ml_sc[0]
        ytest_ml = np.vstack((ytest_ml, a))
        b = ytest_ml[i] + (sigma / 2.0) * (ytest_ml[i+1] - 2.0 * ytest_ml[i] + yf_ml[i - 1])
        yf_ml = np.vstack((yf_ml, b))
        e = a.reshape(1,n-1)
        ytemp = ytest[0]
        ytemp = ytemp.reshape(1,2*(n-1))
        ee = np.concatenate((ytemp,e), axis = 1)
        ee = ee[0,n-1:]
        ytest = ee.reshape(1,2*(n-1))
    
    return ytest_ml

def model_predict_lf4(testing_set, m, n, dt, sc_input, sc_output):
    print("lf4")
    custom_model = load_model('best_model.hd5',custom_objects={'coeff_determination': coeff_determination})
    
    ytest = [testing_set[0], testing_set[1], testing_set[2], testing_set[3]]
    ytest = np.array(ytest)
    ytest = ytest.reshape(1,4*(n-1))
    
    ytest_ml = [testing_set[0]]
    ytest_ml = np.array(ytest_ml)
    ytest_ml = np.vstack((ytest_ml, testing_set[1]))
    ytest_ml = np.vstack((ytest_ml, testing_set[2]))
    ytest_ml = np.vstack((ytest_ml, testing_set[3]))
    
    for i in range(3,m):
        ytest_sc = sc_input.transform(ytest)
        slope_ml = custom_model.predict(ytest_sc) # slope from LSTM/ ML model
        slope_ml_sc = sc_output.inverse_transform(slope_ml)
        a = ytest_ml[i-1] + 2.0*dt*slope_ml_sc[0] # y1 at next time step
        ytest_ml = np.vstack((ytest_ml, a))
        e = a.reshape(1,n-1)
        ytemp = ytest[0]
        ytemp = ytemp.reshape(1,4*(n-1))
        ee = np.concatenate((ytemp,e), axis = 1)
        ee = ee[0,n-1:]
        ytest = ee.reshape(1,4*(n-1))  
    
    return ytest_ml

def model_predict_lff4(testing_set, m, n, dt, sigma, sc_input, sc_output):
    print("lff4")
    custom_model = load_model('best_model.hd5',custom_objects={'coeff_determination': coeff_determination})
    
    ytest = [testing_set[0], testing_set[1], testing_set[2], testing_set[3]]
    ytest = np.array(ytest)
    ytest = ytest.reshape(1,4*(n-1))
    
    ytest_ml = [testing_set[0]]
    ytest_ml = np.array(ytest_ml)
    ytest_ml = np.vstack((ytest_ml, testing_set[1]))
    ytest_ml = np.vstack((ytest_ml, testing_set[2]))
    ytest_ml = np.vstack((ytest_ml, testing_set[3]))
    yf_ml = [testing_set[0]]
    yf_ml = np.array(yf_ml)
    yf_ml = np.vstack((yf_ml, testing_set[1]))
    yf_ml = np.vstack((yf_ml, testing_set[2]))
    
    for i in range(3,m):
        ytest_sc = sc_input.transform(ytest)
        slope_ml = custom_model.predict(ytest_sc) # slope from LSTM/ ML model
        slope_ml_sc = sc_output.inverse_transform(slope_ml)
        a = yf_ml[i-1] + 2.0*dt*slope_ml_sc
        ytest_ml = np.vstack((ytest_ml, a))
        b = ytest_ml[i] + (sigma / 2.0) * (ytest_ml[i+1] - 2.0 * ytest_ml[i] + yf_ml[i - 1])
        yf_ml = np.vstack((yf_ml, b))
        e = a.reshape(1,n-1)
        ytemp = ytest[0]
        ytemp = ytemp.reshape(1,4*(n-1))
        ee = np.concatenate((ytemp,e), axis = 1)
        ee = ee[0,n-1:]
        ytest = ee.reshape(1,4*(n-1))
    
    return ytest_ml


def model_predict_seq1(testing_set, m, n, dt, sc_input, sc_output):
    print("seq1")
    custom_model = load_model('best_model.hd5',custom_objects={'coeff_determination': coeff_determination})
    
    ytest = [testing_set[0]]
    ytest = np.array(ytest)
    
    ytest_ml = [testing_set[0]]
    ytest_ml = np.array(ytest_ml)
    
    for i in range(1,m):
        ytest_sc = sc_input.transform(ytest)
        slope_ml = custom_model.predict(ytest_sc) # slope from LSTM/ ML model
        slope_ml_sc = sc_output.inverse_transform(slope_ml)
        a = slope_ml_sc # y1 at next time step
        ytest_ml = np.vstack((ytest_ml, a))
        ytest = a.reshape(1,1*(n-1))
        
    return ytest_ml


def model_predict_seq2(testing_set, m, n, dt, sc_input, sc_output):
    print("seq2")
    custom_model = load_model('best_model.hd5',custom_objects={'coeff_determination': coeff_determination})

    ytest = [testing_set[0], testing_set[1]]
    ytest = np.array(ytest)
    ytest = ytest.reshape(1,2*(n-1))

    ytest_ml = [testing_set[0]]
    ytest_ml = np.array(ytest_ml)
    ytest_ml = np.vstack((ytest_ml, testing_set[1]))

    for i in range(2,m):
        ytest_sc = sc_input.transform(ytest)
        slope_ml = custom_model.predict(ytest_sc) # slope from LSTM/ ML model
        slope_ml_sc = sc_output.inverse_transform(slope_ml)
        a = slope_ml_sc[0] # y1 at next time step
        ytest_ml = np.vstack((ytest_ml, a))
        e = a.reshape(1,n-1)
        ytemp = ytest[0]
        ytemp = ytemp.reshape(1,2*(n-1))
        ee = np.concatenate((ytemp,e), axis = 1)
        ee = ee[0,n-1:]
        ytest = ee.reshape(1,2*(n-1))

    return ytest_ml

def model_predict_seq4(testing_set, m, n, dt, sc_input, sc_output):
    print("seq4")
    custom_model = load_model('best_model.hd5',custom_objects={'coeff_determination': coeff_determination})

    ytest = [testing_set[0], testing_set[1], testing_set[2], testing_set[3]]
    ytest = np.array(ytest)
    ytest = ytest.reshape(1,4*(n-1))

    ytest_ml = [testing_set[0]]
    ytest_ml = np.array(ytest_ml)
    ytest_ml = np.vstack((ytest_ml, testing_set[1]))
    ytest_ml = np.vstack((ytest_ml, testing_set[2]))
    ytest_ml = np.vstack((ytest_ml, testing_set[3]))

    for i in range(4,m):
        ytest_sc = sc_input.transform(ytest)
        slope_ml = custom_model.predict(ytest_sc) # slope from LSTM/ ML model
        slope_ml_sc = sc_output.inverse_transform(slope_ml)
        a = slope_ml_sc[0] # y1 at next time step
        ytest_ml = np.vstack((ytest_ml, a))
        e = a.reshape(1,n-1)
        ytemp = ytest[0]
        ytemp = ytemp.reshape(1,4*(n-1))
        ee = np.concatenate((ytemp,e), axis = 1)
        ee = ee[0,n-1:]
        ytest = ee.reshape(1,4*(n-1))

    return ytest_ml


def model_predict_e1(testing_set, m, n, dt, sc_input, sc_output):
    print("e1")
    custom_model = load_model('best_model.hd5',custom_objects={'coeff_determination': coeff_determination})

    ytest = [testing_set[0]]
    ytest = np.array(ytest)

    ytest_ml = [testing_set[0]]
    ytest_ml = np.array(ytest_ml)

    for i in range(1,m):
        ytest_sc = sc_input.transform(ytest)
        slope_ml = custom_model.predict(ytest_sc) # slope from LSTM/ ML model
        slope_ml_sc = sc_output.inverse_transform(slope_ml)
        a = ytest_ml[i-1] + 1.0*dt*slope_ml_sc[0] # y1 at next time step
        ytest_ml = np.vstack((ytest_ml, a))
        ytest = a.reshape(1,(n-1))

    return ytest_ml


def model_predict_e2(testing_set, m, n, dt, sc_input, sc_output):
    print("e2")
    custom_model = load_model('best_model.hd5',custom_objects={'coeff_determination': coeff_determination})

    ytest = [testing_set[0], testing_set[1]]
    ytest = np.array(ytest)
    ytest = ytest.reshape(1,2*(n-1))

    ytest_ml = [testing_set[0]]
    ytest_ml = np.array(ytest_ml)
    ytest_ml = np.vstack((ytest_ml, testing_set[1]))

    for i in range(2,m):
        ytest_sc = sc_input.transform(ytest)
        slope_ml = custom_model.predict(ytest_sc) # slope from LSTM/ ML model
        slope_ml_sc = sc_output.inverse_transform(slope_ml)
        a = ytest_ml[i-1] + 1.0*dt*slope_ml_sc[0] # y1 at next time step
        ytest_ml = np.vstack((ytest_ml, a))
        e = a.reshape(1,n-1)
        ytemp = ytest[0]
        ytemp = ytemp.reshape(1,2*(n-1))
        ee = np.concatenate((ytemp,e), axis = 1)
        ee = ee[0,n-1:]
        ytest = ee.reshape(1,2*(n-1))

    return ytest_ml


def model_predict_e4(testing_set, m, n, dt, sc_input, sc_output):
    print("e4")
    custom_model = load_model('best_model.hd5',custom_objects={'coeff_determination': coeff_determination})

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
        a = ytest_ml[i-1] + 1.0*dt*slope_ml_sc[0] # y1 at next time step
        ytest_ml = np.vstack((ytest_ml, a))
        e = a.reshape(1,n-1)
        ytemp = ytest[0]
        ytemp = ytemp.reshape(1,4*(n-1))
        ee = np.concatenate((ytemp,e), axis = 1)
        ee = ee[0,n-1:]
        ytest = ee.reshape(1,4*(n-1))

    return ytest_ml

def model_predict_e1a(testing_set, m, n, dt, sc_input, sc_output):
    print("rn1")
    custom_model = load_model('best_model.hd5',custom_objects={'coeff_determination': coeff_determination})

    ytest = [testing_set[0]]
    ytest = np.array(ytest)

    ytest_ml = [testing_set[0]]
    ytest_ml = np.array(ytest_ml)

    for i in range(1,m):
        ytest_sc = sc_input.transform(ytest)
        slope_ml = custom_model.predict(ytest_sc) # slope from LSTM/ ML model
        slope_ml_sc = sc_output.inverse_transform(slope_ml)
        a = ytest_ml[i-1] + dt*slope_ml_sc[0] # y1 at next time step
        ytest_ml = np.vstack((ytest_ml, a))
        ytest = a.reshape(1,(n-1))

    return ytest_ml

def model_predict_rn1(testing_set, m, n, dt, sc_input, sc_output):
    print("rn1")
    custom_model = load_model('best_model.hd5',custom_objects={'coeff_determination': coeff_determination})

    ytest = [testing_set[0]]
    ytest = np.array(ytest)

    ytest_ml = [testing_set[0]]
    ytest_ml = np.array(ytest_ml)

    for i in range(1,m):
        ytest_sc = sc_input.transform(ytest)
        slope_ml = custom_model.predict(ytest_sc) # slope from LSTM/ ML model
        slope_ml_sc = sc_output.inverse_transform(slope_ml)
        a = ytest_ml[i-1] + slope_ml_sc[0] # y1 at next time step
        ytest_ml = np.vstack((ytest_ml, a))
        ytest = a.reshape(1,(n-1))

    return ytest_ml


def model_predict_rn2(testing_set, m, n, dt, sc_input, sc_output):
    print("rn2")
    custom_model = load_model('best_model.hd5',custom_objects={'coeff_determination': coeff_determination})

    ytest = [testing_set[0], testing_set[1]]
    ytest = np.array(ytest)
    ytest = ytest.reshape(1,2*(n-1))
    
    ytest_ml = [testing_set[0]]
    ytest_ml = np.array(ytest_ml)
    ytest_ml = np.vstack((ytest_ml, testing_set[1]))
    
    for i in range(2,m):
        ytest_sc = sc_input.transform(ytest)
        slope_ml = custom_model.predict(ytest_sc) # slope from LSTM/ ML model
        slope_ml_sc = sc_output.inverse_transform(slope_ml)
        a = ytest_ml[i-1] + slope_ml_sc[0] # y1 at next time step
        ytest_ml = np.vstack((ytest_ml, a))
        e = a.reshape(1,n-1)
        ytemp = ytest[0]
        ytemp = ytemp.reshape(1,2*(n-1))
        ee = np.concatenate((ytemp,e), axis = 1)
        ee = ee[0,n-1:]
        ytest = ee.reshape(1,2*(n-1))

    return ytest_ml


def model_predict_rn4(testing_set, m, n, dt, sc_input, sc_output):
    print("rn4")
    custom_model = load_model('best_model.hd5',custom_objects={'coeff_determination': coeff_determination})

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

def model_predict_test(testing_set, m, n, dt1, legs, slopenet, sigma, sc_input, sc_output):
    print("trial")
    custom_model = load_model('best_model.hd5',custom_objects={'coeff_determination': coeff_determination})
    
    ytest = [testing_set[0]]
    ytest = np.array(ytest)
    
    ytest_ml = [testing_set[0]]
    ytest_ml = np.array(ytest_ml)
    
    for i in range(1,m):
        ytest_sc = sc_input.transform(ytest)
        slope_ml = custom_model.predict(ytest_sc) # slope from LSTM/ ML model
        slope_ml_sc = sc_output.inverse_transform(slope_ml)
        a = ytest_ml[i-1] + slope_ml_sc[0] # y1 at next time step
        ytest_ml = np.vstack((ytest_ml, a))
        ytest = a.reshape(1,(n-1))  
        
    return ytest_ml