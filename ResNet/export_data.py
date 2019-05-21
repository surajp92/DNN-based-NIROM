#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:38:42 2019

@author: Suraj Pawar
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

#%%
def plot_results_rom(dt1, slopenet, legs):
    font = {'family' : 'Times New Roman',
        'size'   : 14}	
    plt.rc('font', **font)
    filename = slopenet+'_p=0'+str(legs)+'.csv'
    dataset_plot = np.genfromtxt(filename, delimiter=",",skip_header=0)
    dataset_gp = np.genfromtxt('./a94gp.csv', delimiter=",",skip_header=0)
    m,n = dataset_plot.shape
    t = dataset_plot[:,0].reshape(m,1)
    t = t-900
    testing_set = dataset_plot[:,1:11]
    gp = dataset_gp[:,1:]
    ytest_ml = dataset_plot[:,11:]
    nrows = 2
    list = [1,5]
    
    fig, axs = plt.subplots(nrows, 1, figsize=(10,4))#, constrained_layout=True)
    for i,j in zip(range(nrows),list):
        #axs[i].plot(ytrains[:,i], color='black', linestyle='-', label=r'$y_'+str(i+1)+'$'+' (True)', zorder=5)
        axs[i].plot(t,testing_set[:,j], color='black', linestyle='-', label=r'$y_'+str(i+1)+'$'+' (True)', zorder=5)
        #axs[i].plot(t,gp[:,j], color='blue', linestyle='-', label=r'$y_'+str(i+1)+'$'+' (True)', zorder=5)
        axs[i].plot(t,ytest_ml[:,j], color='red', linestyle='-', label=r'$y_'+str(i+1)+'$'+' (GP)', zorder=5) 
        axs[i].set_xlim([t[0], t[m-1]])
        #axs[i].set_ylim([min(ytest_ml[:,j])-0.2, max(ytest_ml[:,j])+0.2])
        axs[i].set_ylim([-(max(ytest_ml[:,j])*1.25), (max(ytest_ml[:,j])*1.25)])
        axs[i].set_ylabel('$a_'+'{'+(str(j)+'}'+'$'), fontsize = 14)
        axs[i].axvspan(t[0], t[0]+0.5*(t[m-1]-t[0]), facecolor='0.5', alpha=0.5)
       
    fig.tight_layout() 
    
    fig.subplots_adjust(bottom=0.15)
    
    line_labels = ["True", "GP", "ML"]#, "ML-Train", "ML-Test"]
    plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.1, ncol=6, labelspacing=0.,  prop={'size': 13} ) #bbox_to_anchor=(0.5, 0.0), borderaxespad=0.1, 
    
    savefile = slopenet+'_p=0'+str(legs)+'.eps'
    fig.savefig(savefile)#, bbox_inches = 'tight', pad_inches = 0.01)
    
#%%
def plot_results_lorenz(dt1, slopenet, legs):
    font = {'family' : 'Times New Roman',
        'size'   : 14}	
    plt.rc('font', **font)
    filename = slopenet+'_p=0'+str(legs)+'.csv'
    dataset_plot = np.genfromtxt(filename, delimiter=",",skip_header=0)
    m,n = dataset_plot.shape
    t = dataset_plot[:,0].reshape(m,1)
    states = dataset_plot[:,1:4]
    ytest_ml = dataset_plot[:,4:]
    nrows = 3
    
    fig, axs = plt.subplots(nrows, 1, figsize=(10,6))#, constrained_layout=True)
    for i in range(nrows):
        #axs[i].plot(ytrains[:,i], color='black', linestyle='-', label=r'$y_'+str(i+1)+'$'+' (True)', zorder=5)
        axs[i].plot(t,states[:,i], color='black', linestyle='-', label=r'$y_'+str(i+1)+'$'+' (True)', zorder=5)
        axs[i].plot(t,ytest_ml[:,i], color='blue', linestyle='-.', label=r'$y_'+str(i+1)+'$'+' (GP)', zorder=5) 
        axs[i].set_xlim([t[0], t[m-1]])
        axs[i].set_ylabel('$a_'+'{'+(str(i)+'}'+'$'), fontsize = 14)
        #axs[i].axvspan(00, 12.5, facecolor='0.5', alpha=0.5)
        
    fig.tight_layout() 
    
    fig.subplots_adjust(bottom=0.15)
    
    line_labels = ["True", "ML"]#, "ML-Train", "ML-Test"]
    plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.1, ncol=6, labelspacing=0.,  prop={'size': 13} ) #bbox_to_anchor=(0.5, 0.0), borderaxespad=0.1, 

    
    savefile = slopenet+'_p=0'+str(legs)+'.eps'
    fig.savefig(savefile)#, bbox_inches = 'tight', pad_inches = 0.01)
    
#%%
def plot_results_ko(dt1, slopenet, legs):
    font = {'family' : 'Times New Roman',
        'size'   : 14}	
    plt.rc('font', **font)
    filename = slopenet+'_p=0'+str(legs)+'.csv'
    dataset_plot = np.genfromtxt(filename, delimiter=",",skip_header=0)
    m,n = dataset_plot.shape
    t = dataset_plot[:,0].reshape(m,1)
    states = dataset_plot[:,1:4]
    ytest_ml = dataset_plot[:,4:]
    for i in range(3):
        plt.figure()    
        plt.plot(states[:,i], 'b-', label=r'$y_'+str(i+1)+'$'+' (True)') 
        plt.plot(ytest_ml[:,i], 'r-', label=r'$y_'+str(i+1)+'$'+' (ML)') 
        plt.ylabel('Response')
        plt.xlabel('Time')
        plt.legend(loc='best')
        name = 'a='+str(i+1)+'.eps'
        #plt.savefig(name, dpi = 400)
        plt.show()
    
#%%     
def calculate_l2norm(ytest_ml, testing_set, legs, slopenet, problem):
    # calculates L2 norm of each series
    m,n = testing_set.shape
    error = testing_set-ytest_ml
    l2norm_sum = 0.0
    l2norm_true = 0.0
    l2norm_nd = 0.0
    for i in range(n-1):
        l2norm_sum += LA.norm(error[:,i])
        
    for i in range(n-1):
        l2norm_true += LA.norm(testing_set[:,i])
    
    l2norm_nd = l2norm_sum/l2norm_true
    
    rmse = l2norm_sum/np.sqrt(m)
    
    list = [legs, slopenet, problem, l2norm_nd, l2norm_sum, l2norm_true, rmse]
    with open('l2norm.csv', 'a') as f:
        f.write("\n")
        for item in list:
            f.write("%s\t" % item)
        
    return l2norm_sum, l2norm_nd

#%%
def export_results(ytest_ml, testing_set, t, slopenet, legs):
    # export result in x y format for further plotting
    filename = slopenet+'_p=0'+str(legs)+'.csv'
    m,n = testing_set.shape
    t = t.reshape(m,1)
    results = np.hstack((t, testing_set, ytest_ml))
    np.savetxt(filename, results, delimiter=",")
    
