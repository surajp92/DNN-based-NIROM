#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 17:16:33 2019

@author: Suraj Pawar
"""
import numpy as np
import matplotlib.pyplot as plt
#plt.rcParams["font.family"] = "Times New Roman"

font = {'family' : 'Times New Roman',
        'size'   : 12}	
plt.rc('font', **font)

import os
#directory = os.path.join("/media/user1/My4TBHD1/Suraj/06_CFDLab_Codes/ODE_ANN/Paper results/ROM results/ROM_Results_Data/Scaled results/B series")
directory = os.path.join("../A series")

gp_solution = np.loadtxt(open('d_pod_GP/d_pod_a.csv', "rb"), delimiter=",", skiprows=0)
ygp = gp_solution[:2000,1:]

#for root,dirs,files in os.walk(directory):
#    for file in files:
#       if file.endswith(".csv"):
file = 'seq4.csv'
filename = file[:-4]
name = filename + 'a.eps'
solution = np.loadtxt(open(file, "rb"), delimiter=",", skiprows=0)
m,n = solution.shape
time = solution[:,0]
time_test = solution[0:1000,0]
time_pred = solution[1000:,0]
time = time-900
time_test = time_test-900
time_pred = time_pred-900
ytrue = solution[:,1:11]
ytestplot = solution[0:1000,11:]
ypredplot = solution[1000:,11:]
   
nrows = 10  ## rows of subplot
ncols = 1  ## colmns of subplot
fig, axs = plt.subplots(nrows, ncols, sharex=True,  figsize=(12,10))
fig.subplots_adjust(hspace=0.4)  
           
           
for i in range(0,nrows):           
    axs[i].plot(time,ytrue[:,i], 'r-', label=r'$y_'+str(i+1)+'$'+' (True)')
    axs[i].plot(time,ygp[:,i], color='orange', label=r'$y_'+str(i+1)+'$'+' (GP)')
    axs[i].plot(time_test,ytestplot[:,i], 'b--', label=r'$y_'+str(i+1)+'$'+' (ML Test)')
    axs[i].plot(time_pred,ypredplot[:,i], 'g--', label=r'$y_'+str(1+1)+'$'+' (ML Pred)')
    axs[i].set_xlim([0, 100])
    axs[i].set_ylabel('$a_'+'{'+(str(i+1)+'}'+'$'), fontsize = 14)
    axs[i].yaxis.set_major_locator(plt.MaxNLocator(3))
    axs[i].yaxis.set_label_coords(-0.07, 0.5)       
   
line_labels = ["True", "ROM-G", "ML-Test", "ML-Pred"]
plt.figlegend( line_labels,  loc = 'lower center', bbox_to_anchor=(0.49, 0.01), ncol=5, labelspacing=0.,  prop={'size': 14} ) # loc = 'lower center', bbox_to_anchor=(0.5, -0.05),
   
fig.text(0.5, 0.08, 'Time(s)', ha='center', fontsize = 14)
fig.text(0.025, 0.5, 'Response', va='center', fontsize = 14, rotation=90)
   
fig.savefig(name, bbox_inches = 'tight')#, bbox_extra_artists=(lgd,), bbox_inches='tight')#, )#, pad_inches = 0)

