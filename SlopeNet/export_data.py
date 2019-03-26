#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:38:42 2019

@author: Suraj Pawar
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_results(ytest_ml, testing_set, m, n):

    for i in range(n-1):
        plt.figure()    
        plt.plot(ytest_ml[:,i], 'b-', label=r'$y_'+str(i+1)+'$'+'ML') 
        plt.plot(testing_set[:,i], 'r-', label=r'$y_'+str(i+1)+'$') 
        plt.ylabel('response ML')
        plt.xlabel('time ML')
        plt.legend(loc='best')
        name = 'dns_bdf_twoleg_a='+str(i+1)+'.eps'
        #plt.savefig(name, dpi = 600)
        plt.show()

def calculate_l2norm(ytest_ml, testing_set, m, n):
    # calculates L2 norm of each series
    print("blank")
    
def export_results(ytest_ml, testing_set, m, n):
    # export result in x y format for further plotting
    print("blank")