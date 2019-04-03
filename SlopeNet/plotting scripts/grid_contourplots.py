#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:17:27 2019

@author: Suraj Pawar
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from matplotlib import ticker, cm

font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 14}	
plt.rc('font', **font)

nx = 128
ny = 1024
nr = 10
#%%
tm_true = pd.read_csv('./true.csv', sep=",", skiprows=0, header=None)
tm_true = np.array(tm_true)
x = tm_true[:, 0]
y = tm_true[:, 1]
z = tm_true[:, 2]
xx = x[:nx+1]
yy = y.reshape((nx+1,ny+1),order='F')[0, :]
tm_true = z.reshape((nx+1,ny+1),order='F')

#X, Y = np.meshgrid(xx, yy)
#Z = tm.reshape((1025,129))

#%%
tm_romgp = pd.read_csv('./romgp.csv', sep=",", skiprows=0, header=None)
tm_romgp = np.array(tm_romgp)
z = tm_romgp[:, 2]
tm_romgp = z.reshape((nx+1,ny+1),order='F')

#%%
tm_s1 = pd.read_csv('./s1.csv', sep=",", skiprows=0, header=None)
tm_s1= np.array(tm_s1)
z = tm_s1[:, 2]
tm_s1 = z.reshape((nx+1,ny+1),order='F')

#%%
tm_e1 = pd.read_csv('./e1.csv', sep=",", skiprows=0, header=None)
tm_e1= np.array(tm_e1)
z = tm_e1[:, 2]
tm_e1 = z.reshape((nx+1,ny+1),order='F')

#%%
tm_l1 = pd.read_csv('./l1.csv', sep=",", skiprows=0, header=None)
tm_l1= np.array(tm_l1)
z = tm_l1[:, 2]
tm_l1 = z.reshape((nx+1,ny+1),order='F')

#%%
tm_b21 = pd.read_csv('./b21.csv', sep=",", skiprows=0, header=None)
tm_b21 = np.array(tm_b21 )
z = tm_b21 [:, 2]
tm_b21  = z.reshape((nx+1,ny+1),order='F')

#%%
tm_b31 = pd.read_csv('./b31.csv', sep=",", skiprows=0, header=None)
tm_b31 = np.array(tm_b31 )
z = tm_b31 [:, 2]
tm_b31  = z.reshape((nx+1,ny+1),order='F')

#%%
fig, axs = plt.subplots(1,7,sharey=True,figsize=(12,10))
fig.subplots_adjust(left= 0.1, right=0.9, bottom = 0.1, top = 0.9, wspace=0.2) 
cs = axs[0].contour(xx,yy,tm_true.T, 40, cmap='inferno', interpolation='bilinear')
axs[0].text(0.4, -0.04, '(a)', transform=axs[0].transAxes, fontsize=16, fontweight='bold', va='top')
axs[0].xaxis.set_major_locator(plt.MaxNLocator(1))

cs = axs[1].contour(xx,yy,tm_romgp.T, 40, cmap='inferno', interpolation='bilinear')
axs[1].text(0.4, -0.04, '(b)', transform=axs[1].transAxes, fontsize=16, fontweight='bold', va='top')
axs[1].xaxis.set_major_locator(plt.MaxNLocator(1))

cs = axs[2].contour(xx,yy,tm_s1.T, 40, cmap='inferno', interpolation='bilinear')
axs[2].text(0.4, -0.04, '(c)', transform=axs[2].transAxes, fontsize=16, fontweight='bold', va='top')
axs[2].xaxis.set_major_locator(plt.MaxNLocator(1))

cs = axs[3].contour(xx,yy,tm_e1.T, 40, cmap='inferno', interpolation='bilinear')
axs[3].text(0.4, -0.04, '(d)', transform=axs[3].transAxes, fontsize=16, fontweight='bold', va='top')
axs[3].xaxis.set_major_locator(plt.MaxNLocator(1))

cs = axs[4].contour(xx,yy,tm_l1.T, 40, cmap='inferno', interpolation='bilinear')
axs[4].text(0.4, -0.04, '(e)', transform=axs[4].transAxes, fontsize=16, fontweight='bold', va='top')
axs[4].xaxis.set_major_locator(plt.MaxNLocator(1))

cs = axs[5].contour(xx,yy,tm_b21.T, 40, cmap='inferno', interpolation='bilinear')
axs[5].text(0.4, -0.04, '(f)', transform=axs[5].transAxes, fontsize=16, fontweight='bold', va='top')
axs[5].xaxis.set_major_locator(plt.MaxNLocator(1))

cs = axs[6].contour(xx,yy,tm_b31.T, 40, cmap='inferno', interpolation='bilinear')
axs[6].text(0.4, -0.04, '(g)', transform=axs[6].transAxes, fontsize=16, fontweight='bold', va='top')
axs[6].xaxis.set_major_locator(plt.MaxNLocator(1))
fig.tight_layout() 

fig.subplots_adjust(bottom=0.08)

cbar_ax = fig.add_axes([0.22, -0.05, 0.6, 0.04])
fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
plt.show()

fig.savefig("temp_oneleg.eps", bbox_inches = 'tight')

#fig, axs = plt.subplots(1,4,figsize=(8,16))
#cs = axs[0].imshow(tm.T, cmap='jet', interpolation='none', origin='lower',extent=[0, 1, 0, 8],  aspect='auto')
#cs = axs[1].imshow(tm.T, cmap='jet', interpolation='none', origin='lower',extent=[0, 1, 0, 8],  aspect='auto')
##cbar = fig.colorbar(cs)
#plt.show()
#
#
#fig, ax = plt.subplots(figsize=(2,16))
#im = plt.imshow(tm.T, cmap='jet', origin='lower',extent=[0, 1, 0, 8],  aspect='auto') #
#plt.colorbar(im, orientation='vertical')
#plt.show()

