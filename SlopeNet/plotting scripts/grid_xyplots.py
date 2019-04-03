#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 21:12:35 2019

@author: Suraj Pawar
"""
import numpy as np
import matplotlib.pyplot as plt
#plt.rcParams["font.family"] = "Times New Roman"
#plt.rcParams.update({'font.size': 12})

font = {'family' : 'Times New Roman',
        'size'   : 14}	
plt.rc('font', **font)


t = np.linspace(0,10, num=200)
fig, axs = plt.subplots(5, 3, figsize=(12,15))#, constrained_layout=True)
i = 3 # i = 0 for x = 0.25; i = 3 for x = -0.25
j = 9 # j = 6 for x = 0.25; j = 9 for x = -0.25
#%%
solution = np.loadtxt(open("sequential_legs=01.csv", "rb"), delimiter=",", skiprows=0)
axs[0, 0].plot(t, solution[:,i],'r-')
axs[0, 0].plot(t, solution[:,i+1],'g-')
axs[0, 0].plot(t, solution[:,i+2],'b-')
axs[0, 0].plot(t, solution[:,j],'r--')
axs[0, 0].plot(t, solution[:,j+1],'g--')
axs[0, 0].plot(t, solution[:,j+2],'b--')
axs[0, 0].set_title('DNN-S (p = 1)', fontsize=14)
axs[0, 0].set_xlim([0, 10])

#%%
solution = np.loadtxt(open("sequential_legs=02.csv", "rb"), delimiter=",", skiprows=0)
axs[0, 1].plot(t, solution[:,i],'r-')
axs[0, 1].plot(t, solution[:,i+1],'g-')
axs[0, 1].plot(t, solution[:,i+2],'b-')
axs[0, 1].plot(t, solution[:,j],'r--')
axs[0, 1].plot(t, solution[:,j+1],'g--')
axs[0, 1].plot(t, solution[:,j+2],'b--')
axs[0, 1].set_title('DNN-S (p = 2)', fontsize=14)
axs[0, 1].set_xlim([0, 10])

#%%
solution = np.loadtxt(open("sequential_legs=04.csv", "rb"), delimiter=",", skiprows=0)
axs[0, 2].plot(t, solution[:,0],'r-')
axs[0, 2].plot(t, solution[:,i+1],'g-')
axs[0, 2].plot(t, solution[:,i+2],'b-')
axs[0, 2].plot(t, solution[:,j],'r--')
axs[0, 2].plot(t, solution[:,j+1],'g--')
axs[0, 2].plot(t, solution[:,j+2],'b--')
axs[0, 2].set_title('DNN-S (p = 4)', fontsize=14)
axs[0, 2].set_xlim([0, 10])

#%%
solution = np.loadtxt(open("euler_legs=01.csv", "rb"), delimiter=",", skiprows=0)
axs[1, 0].plot(t, solution[:,i],'r-')
axs[1, 0].plot(t, solution[:,i+1],'g-')
axs[1, 0].plot(t, solution[:,i+2],'b-')
axs[1, 0].plot(t, solution[:,j],'r--')
axs[1, 0].plot(t, solution[:,j+1],'g--')
axs[1, 0].plot(t, solution[:,j+2],'b--')
axs[1, 0].set_title('SlopeNet-E (p = 1)', fontsize=14)
axs[1, 0].set_xlim([0, 10])

#%%
solution = np.loadtxt(open("euler_legs=02.csv", "rb"), delimiter=",", skiprows=0)
axs[1, 1].plot(t, solution[:,i],'r-')
axs[1, 1].plot(t, solution[:,i+1],'g-')
axs[1, 1].plot(t, solution[:,i+2],'b-')
axs[1, 1].plot(t, solution[:,j],'r--')
axs[1, 1].plot(t, solution[:,j+1],'g--')
axs[1, 1].plot(t, solution[:,j+2],'b--')
axs[1, 1].set_title('SlopeNet-E (p = 2)', fontsize=14)
axs[1, 1].set_xlim([0, 10])

#%%
solution = np.loadtxt(open("euler_legs=04.csv", "rb"), delimiter=",", skiprows=0)
axs[1, 2].plot(t, solution[:,i],'r-')
axs[1, 2].plot(t, solution[:,i+1],'g-')
axs[1, 2].plot(t, solution[:,i+2],'b-')
axs[1, 2].plot(t, solution[:,j],'r--')
axs[1, 2].plot(t, solution[:,j+1],'g--')
axs[1, 2].plot(t, solution[:,j+2],'b--')
axs[1, 2].set_title('SlopeNet-E (p = 4)', fontsize=14)
axs[1, 2].set_xlim([0, 10])

#%%
solution = np.loadtxt(open("leapfrog_legs=01.csv", "rb"), delimiter=",", skiprows=0)
axs[2, 0].plot(t, solution[:,i],'r-')
axs[2, 0].plot(t, solution[:,i+1],'g-')
axs[2, 0].plot(t, solution[:,i+2],'b-')
axs[2, 0].plot(t, solution[:,j],'r--')
axs[2, 0].plot(t, solution[:,j+1],'g--')
axs[2, 0].plot(t, solution[:,j+2],'b--')
axs[2, 0].set_title('SlopeNet-L (p = 1)', fontsize=14)
axs[2, 0].set_xlim([0, 10])
axs[2,0].yaxis.set_label_coords(-0.15, 0.5)

#%%
solution = np.loadtxt(open("leapfrog_legs=02.csv", "rb"), delimiter=",", skiprows=0)
axs[2, 1].plot(t, solution[:,i],'r-')
axs[2, 1].plot(t, solution[:,i+1],'g-')
axs[2, 1].plot(t, solution[:,i+2],'b-')
axs[2, 1].plot(t, solution[:,j],'r--')
axs[2, 1].plot(t, solution[:,j+1],'g--')
axs[2, 1].plot(t, solution[:,j+2],'b--')
axs[2, 1].set_title('SlopeNet-L (p = 2)', fontsize=14)
axs[2, 1].set_xlim([0, 10])

#%%
solution = np.loadtxt(open("leapfrog_legs=04.csv", "rb"), delimiter=",", skiprows=0)
axs[2, 2].plot(t, solution[:,i],'r-')
axs[2, 2].plot(t, solution[:,i+1],'g-')
axs[2, 2].plot(t, solution[:,i+2],'b-')
axs[2, 2].plot(t, solution[:,j],'r--')
axs[2, 2].plot(t, solution[:,j+1],'g--')
axs[2, 2].plot(t, solution[:,j+2],'b--')
axs[2, 2].set_title('SlopeNet-L (p = 4)', fontsize=14)
axs[2, 2].set_xlim([0, 10])

#%%
solution = np.loadtxt(open("bdf2_legs=01.csv", "rb"), delimiter=",", skiprows=0)
axs[3, 0].plot(t, solution[:,i],'r-')
axs[3, 0].plot(t, solution[:,i+1],'g-')
axs[3, 0].plot(t, solution[:,i+2],'b-')
axs[3, 0].plot(t, solution[:,j],'r--')
axs[3, 0].plot(t, solution[:,j+1],'g--')
axs[3, 0].plot(t, solution[:,j+2],'b--')
axs[3, 0].set_title('SlopeNet-B2 (p = 1)', fontsize=14)
axs[3, 0].set_xlim([0, 10])
axs[3,0].yaxis.set_label_coords(-0.15, 0.5)

#%%
solution = np.loadtxt(open("bdf2_legs=02.csv", "rb"), delimiter=",", skiprows=0)
axs[3, 1].plot(t, solution[:,i],'r-')
axs[3, 1].plot(t, solution[:,i+1],'g-')
axs[3, 1].plot(t, solution[:,i+2],'b-')
axs[3, 1].plot(t, solution[:,j],'r--')
axs[3, 1].plot(t, solution[:,j+1],'g--')
axs[3, 1].plot(t, solution[:,j+2],'b--')
axs[3, 1].set_title('SlopeNet-B2 (p = 2)', fontsize=14)
axs[3, 1].set_xlim([0, 10])

#%%
solution = np.loadtxt(open("bdf2_legs=04.csv", "rb"), delimiter=",", skiprows=0)
axs[3, 2].plot(t, solution[:,i],'r-')
axs[3, 2].plot(t, solution[:,i+1],'g-')
axs[3, 2].plot(t, solution[:,i+2],'b-')
axs[3, 2].plot(t, solution[:,j],'r--')
axs[3, 2].plot(t, solution[:,j+1],'g--')
axs[3, 2].plot(t, solution[:,j+2],'b--')
axs[3, 2].set_title('SlopeNet-B2 (p = 4)', fontsize=14)
axs[3, 2].set_xlim([0, 10])

#%%
solution = np.loadtxt(open("bdf3_legs=01.csv", "rb"), delimiter=",", skiprows=0)
axs[4, 0].plot(t, solution[:,i],'r-')
axs[4, 0].plot(t, solution[:,i+1],'g-')
axs[4, 0].plot(t, solution[:,i+2],'b-')
axs[4, 0].plot(t, solution[:,j],'r--')
axs[4, 0].plot(t, solution[:,j+1],'g--')
axs[4, 0].plot(t, solution[:,j+2],'b--')
axs[4, 0].set_title('SlopeNet-B3 (p = 1)', fontsize=14)
axs[4, 0].set_xlim([0, 10])
axs[4,0].yaxis.set_label_coords(-0.15, 0.5)

#%%
solution = np.loadtxt(open("bdf3_legs=02.csv", "rb"), delimiter=",", skiprows=0)
axs[4, 1].plot(t, solution[:,i],'r-')
axs[4, 1].plot(t, solution[:,i+1],'g-')
axs[4, 1].plot(t, solution[:,i+2],'b-')
axs[4, 1].plot(t, solution[:,j],'r--')
axs[4, 1].plot(t, solution[:,j+1],'g--')
axs[4, 1].plot(t, solution[:,j+2],'b--')
axs[4, 1].set_title('SlopeNet-B3 (p = 2)', fontsize=14)
axs[4, 1].set_xlim([0, 10])

#%%
solution = np.loadtxt(open("bdf3_legs=04.csv", "rb"), delimiter=",", skiprows=0)
axs[4, 2].plot(t, solution[:,i],'r-')
axs[4, 2].plot(t, solution[:,i+1],'g-')
axs[4, 2].plot(t, solution[:,i+2],'b-')
axs[4, 2].plot(t, solution[:,j],'r--')
axs[4, 2].plot(t, solution[:,j+1],'g--')
axs[4, 2].plot(t, solution[:,j+2],'b--')
axs[4, 2].set_title('SlopeNet-B3 (p = 4)', fontsize=14)
axs[4, 2].set_xlim([0, 10])

#%%
#solution = np.loadtxt(open("bdf4_legs=01.csv", "rb"), delimiter=",", skiprows=0)
#axs[5, 0].plot(t, solution[:,i],'r-')
#axs[5, 0].plot(t, solution[:,i+1],'g-')
#axs[5, 0].plot(t, solution[:,i+2],'b-')
#axs[5, 0].plot(t, solution[:,j],'r--')
#axs[5, 0].plot(t, solution[:,j+1],'g--')
#axs[5, 0].plot(t, solution[:,j+2],'b--')
#axs[5, 0].set_title('SlopeNet-B4 (p = 1)', fontsize=14)
#axs[5, 0].set_xlim([0, 10])
#axs[5,0].yaxis.set_label_coords(-0.15, 0.5)

#%%
#solution = np.loadtxt(open("bdf4_legs=02.csv", "rb"), delimiter=",", skiprows=0)
#axs[5, 1].plot(t, solution[:,i],'r-')
#axs[5, 1].plot(t, solution[:,i+1],'g-')
#axs[5, 1].plot(t, solution[:,i+2],'b-')
#axs[5, 1].plot(t, solution[:,j],'r--')
#axs[5, 1].plot(t, solution[:,j+1],'g--')
#axs[5, 1].plot(t, solution[:,j+2],'b--')
#axs[5, 1].set_title('SlopeNet-B4 (p = 2)', fontsize=14)
#axs[5, 1].set_xlim([0, 10])

#%%
#solution = np.loadtxt(open("bdf4_legs=04.csv", "rb"), delimiter=",", skiprows=0)
#axs[5, 2].plot(t, solution[:,i],'r-')
#axs[5, 2].plot(t, solution[:,i+1],'g-')
#axs[5, 2].plot(t, solution[:,i+2],'b-')
#axs[5, 2].plot(t, solution[:,j],'r--')
#axs[5, 2].plot(t, solution[:,j+1],'g--')
#axs[5, 2].plot(t, solution[:,j+2],'b--')
#axs[5, 2].set_title('SlopeNet-B4 (p = 4)', fontsize=14)
#axs[5, 2].set_xlim([0, 10])

for ax in axs.flat:
    ax.set(xlabel='Time', ylabel='Response')
    
for ax in axs.flat:
    ax.label_outer()

fig.tight_layout() 

fig.subplots_adjust(bottom=0.06)

line_labels = ["$y_1$ (True)","$y_2$ (True)", "$y_3$ (True)", "$y_1$ (ML-Pred)","$y_2$ (ML-Pred)", "$y_3$ (ML-Pred)"]
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.1, ncol=6, labelspacing=0.,  prop={'size': 13} ) #bbox_to_anchor=(0.5, 0.0), borderaxespad=0.1, 



fig.savefig('fig32.pdf')#, bbox_inches = 'tight', pad_inches = 0.01)