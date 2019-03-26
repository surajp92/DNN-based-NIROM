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

def coeff_determination(y_true, y_pred):
        SS_res =  K.sum(K.square( y_true-y_pred ))
        SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
        return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def model_predict(testing_set, m, n, dt, legs, slopnet):
    if (legs == 2) & (slopnet == "BDF"):
        ytest_ml = model_predict_bdf2(testing_set, m, n, dt)
        return ytest_ml
    elif (legs == 3) & (slopnet == "BDF"):
        ytest_ml = model_predict_bdf3(testing_set, m, n, dt)
        return ytest_ml
    elif (legs == 4) & (slopnet == "BDF"):
        ytest_ml = model_predict_bdf4(testing_set, m, n, dt)
        return ytest_ml
    elif (legs == 1) & (slopnet == "LEAPFROG"):
        ytest_ml = model_predict_lf1(testing_set, m, n, dt)
        return ytest_ml
    elif (legs == 2) & (slopnet == "LEAPFROG"):
        ytest_ml = model_predict_lf2(testing_set, m, n, dt)
        return ytest_ml
    elif (legs == 4) & (slopnet == "LEAPFROG"):
        ytest_ml = model_predict_lf4(testing_set, m, n, dt)
        return ytest_ml
    elif (legs == 1) & (slopnet == "SEQ"):
        ytest_ml = model_predict_seq1(testing_set, m, n, dt)
        return ytest_ml
    
def model_predict_bdf2(testing_set, m, n, dt):
    print("bdf2")
    custom_model = load_model('best_model.hd5',custom_objects={'coeff_determination': coeff_determination})
    
    # create input at t= 0 for the model testing
    ytest = [testing_set[0], testing_set[1]]
    ytest = np.array(ytest)
    ytest = ytest.reshape(1,20)
    
    ytest_ml = [testing_set[0]]
    ytest_ml = np.array(ytest_ml)
    ytest_ml = np.vstack((ytest_ml, testing_set[1]))
      
    for i in range(2,testing_set.shape[0]):
        slope_ml = custom_model.predict(ytest) # slope from LSTM/ ML model
        a = (4.0/3.0)*ytest_ml[i-1]- (1.0/3.0)*ytest_ml[i-2] + (2.0/3.0)*dt*slope_ml[0] # y1 at next time step
        ytest_ml = np.vstack((ytest_ml, a))
        e = a.reshape(1,n-1)
        ytemp = ytest[0]
        ytemp = ytemp.reshape(1,2*(n-1))
        ee = np.concatenate((ytemp,e), axis = 1)
        ee = ee[0,10:]
        ytest = ee.reshape(1,2*(n-1)) #np.vstack((ytest, ee)) # add [y1, y2, y3] at (n+1) to input test for next slope prediction
    
    return ytest_ml

def model_predict_bdf3(testing_set, m, n, dt):
    print("bdf3")
    custom_model = load_model('best_model.hd5',custom_objects={'coeff_determination': coeff_determination})
    
    # create input at t= 0 for the model testing
    ytest = [testing_set[0], testing_set[1], testing_set[2]]
    ytest = np.array(ytest)
    ytest = ytest.reshape(1,3*(n-1))
    
    ytest_ml = [testing_set[0]]
    ytest_ml = np.array(ytest_ml)
    ytest_ml = np.vstack((ytest_ml, testing_set[1]))
    ytest_ml = np.vstack((ytest_ml, testing_set[2]))
    
    for i in range(3,testing_set.shape[0]):
        slope_ml = custom_model.predict(ytest) # slope from LSTM/ ML model
        a = (18.0/11.0)*ytest_ml[i-1]- (9.0/11.0)*ytest_ml[i-2]+(2.0/11.0)* ytest_ml[i-3]+ (6.0/11.0)*dt*slope_ml[0] # y1 at next time step
        ytest_ml = np.vstack((ytest_ml, a))
        e = a.reshape(1,n-1)
        ytemp = ytest[0]
        ytemp = ytemp.reshape(1,3*(n-1))
        ee = np.concatenate((ytemp,e), axis = 1)
        ee = ee[0,10:]
        ytest = ee.reshape(1,3*(n-1)) #np.vstack((ytest, ee)) # add [y1, y2, y3] at (n+1) to input test for next slope prediction
        
    return ytest_ml

def model_predict_bdf4(testing_set, m, n, dt):
    print("bdf4")
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
    
    for i in range(4,testing_set.shape[0]):
        slope_ml = custom_model.predict(ytest) # slope from LSTM/ ML model
        a = (48.0/25.0)*ytest_ml[i-1]- (36.0/25.0)*ytest_ml[i-2]+(16.0/25.0)*ytest_ml[i-3]-(3.0/25.0)*ytest_ml[i-4]+(12.0/25.0)*dt*slope_ml[0] # y1 at next time step
        ytest_ml = np.vstack((ytest_ml, a))
        e = a.reshape(1,n-1)
        ytemp = ytest[0]
        ytemp = ytemp.reshape(1,4*(n-1))
        ee = np.concatenate((ytemp,e), axis = 1)
        ee = ee[0,10:]
        ytest = ee.reshape(1,4*(n-1)) 
    
    return ytest_ml

def model_predict_lf1(testing_set, m, n, dt):
    print("lf1")
    custom_model = load_model('best_model.hd5',custom_objects={'coeff_determination': coeff_determination})
    
    # create input at t= 0 for the model testing
    ytest = [testing_set[0]]
    ytest = np.array(ytest)
    
    ytest_ml = [testing_set[0]]
    ytest_ml = np.array(ytest_ml)
    ytest_ml = np.vstack((ytest_ml, testing_set[1]))
    
    for i in range(2,testing_set.shape[0]-1):
        slope_ml = custom_model.predict(ytest) # slope from LSTM/ ML model
        a = ytest_ml[i-2] + 2.0*dt*slope_ml[0] # y1 at next time step
        ytest_ml = np.vstack((ytest_ml, a))
        ytest = a.reshape(1,(n-1)) 
        
    return ytest_ml
    
    
def model_predict_lf2(testing_set, m, n, dt):
    print("lf2")
    custom_model = load_model('best_model.hd5',custom_objects={'coeff_determination': coeff_determination})
    
    ytest = [testing_set[0], testing_set[1]]
    ytest = np.array(ytest)
    ytest = ytest.reshape(1,2*(n-1))
    
    ytest_ml = [testing_set[0]]
    ytest_ml = np.array(ytest_ml)
    ytest_ml = np.vstack((ytest_ml, testing_set[1]))
    
    for i in range(2,testing_set.shape[0]):
        slope_ml = custom_model.predict(ytest) # slope from LSTM/ ML model
        a = ytest_ml[i-2] + 2.0*dt*slope_ml[0] # y1 at next time step
        ytest_ml = np.vstack((ytest_ml, a))
        e = a.reshape(1,n-1)
        ytemp = ytest[0]
        ytemp = ytemp.reshape(1,2*(n-1))
        ee = np.concatenate((ytemp,e), axis = 1)
        ee = ee[0,10:]
        ytest = ee.reshape(1,2*(n-1))     
    
    return ytest_ml

def model_predict_lf4(testing_set, m, n, dt):
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
    
    for i in range(4,testing_set.shape[0]):
        slope_ml = custom_model.predict(ytest) # slope from LSTM/ ML model
        a = ytest_ml[i-2] + 2.0*dt*slope_ml[0] # y1 at next time step
        ytest_ml = np.vstack((ytest_ml, a))
        e = a.reshape(1,n-1)
        ytemp = ytest[0]
        ytemp = ytemp.reshape(1,4*(n-1))
        ee = np.concatenate((ytemp,e), axis = 1)
        ee = ee[0,10:]
        ytest = ee.reshape(1,4*(n-1))  #np.vstack((ytest, ee)) # add [y1, y2, y3] at (n+1) to input test for next slope prediction
    
    return ytest_ml

def model_predict_seq1(testing_set, m, n, dt):
    print("seq1")
    custom_model = load_model('best_model.hd5',custom_objects={'coeff_determination': coeff_determination})
    
    ytest = [testing_set[0]]
    ytest = np.array(ytest)
    
    ytest_ml = [testing_set[0]]
    ytest_ml = np.array(ytest_ml)
    
    for i in range(1,testing_set.shape[0]-1):
        slope_ml = custom_model.predict(ytest) # slope from LSTM/ ML model
        a = 1.0*slope_ml[0] # y1 at next time step
        ytest_ml = np.vstack((ytest_ml, a))
        ytest = a.reshape(1,1*(n-1)) 
        
    return ytest_ml
    