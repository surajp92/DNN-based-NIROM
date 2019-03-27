#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:38:42 2019

@author: Suraj Pawar
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

def plot_results():
    solution = np.loadtxt(open("solution.csv", "rb"), delimiter=",", skiprows=0)
    m,n = solution.shape
    for i in range(int(n/2)):
        plt.figure()    
        plt.plot(solution[:,i], 'b-', label=r'$y_'+str(i+1)+'$'+' (True)') 
        plt.plot(solution[:,i+int(n/2)], 'r-', label=r'$y_'+str(i+1)+'$'+' (ML)') 
        plt.ylabel('Response')
        plt.xlabel('Time')
        plt.legend(loc='best')
        name = 'dns_bdf_twoleg_a='+str(i+1)+'.eps'
        #plt.savefig(name, dpi = 600)
        plt.show()


def calculate_l2norm(ytest_ml, testing_set, m, n):
    # calculates L2 norm of each series
    error = testing_set-ytest_ml
    l2norm_sum = 0.0
    l2norm_true = 0.0
    l2norm_nd = 0.0
    for i in range(n-1):
        l2norm_sum += LA.norm(error[:,i])
        
    for i in range(n-1):
        l2norm_true += LA.norm(testing_set[:,i])
    
    l2norm_nd = l2norm_sum/l2norm_true
        
    return l2norm_sum, l2norm_nd

def export_results(ytest_ml, testing_set, m, n):
    # export result in x y format for further plotting
    results = np.hstack((testing_set, ytest_ml))
    np.savetxt("solution.csv", results, delimiter=",")
    
