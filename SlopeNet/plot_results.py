#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 17:16:33 2019

@author: Suraj Pawar
"""
import numpy as np
import matplotlib.pyplot as plt

solution = np.loadtxt(open("solution.csv", "rb"), delimiter=",", skiprows=0)
m,n = solution.shape
time = solution[:,0]
time_test = solution[0:1000,0]
time_pred = solution[1000:,0]
ytrue = solution[:,1:11]
ytestplot = solution[0:1000,11:]
ypredplot = solution[1000:,11:]
for i in range(int(n/2)):
    plt.figure()    
    plt.plot(time,ytrue[:,i], 'r-', label=r'$y_'+str(i+1)+'$'+' (True)')
    plt.plot(time_test,ytestplot[:,i], 'b-', label=r'$y_'+str(i+1)+'$'+' (ML Test)')
    plt.plot(time_pred,ypredplot[:,i], 'g-', label=r'$y_'+str(1+1)+'$'+' (ML Pred)')
    #plt.plot(solution[:,i+int(n/2)], 'r-', label=r'$y_'+str(i+1)+'$'+' (ML)') 
    plt.ylabel('Response')
    plt.xlabel('Time')
    plt.legend(loc='best')
    name = 'a='+str(i+1)+'.eps'
    #plt.savefig(name, dpi = 400)
    plt.show()