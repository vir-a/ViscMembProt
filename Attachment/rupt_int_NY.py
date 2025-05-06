#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 14:11:57 2023

@author: vira
"""

%reset -f
import os
import importlib
filedir=os.path.dirname(os.path.abspath(__file__))
os.chdir(filedir)
import fluktrupfunk as fxn
importlib.reload(fxn)#%%#%%
#%%
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import csv
from decimal import Decimal
from scipy.signal import argrelextrema
#%%
filedir=os.path.dirname(os.path.abspath(__file__))
os.chdir(filedir)


folder='int_varQ'
path1 = os.path.join(os.getcwd(),folder)
if not os.path.exists(path1):
    os.mkdir(path1)
os.chdir(folder)
cwd0 = os.getcwd()
#%%
# =============================================================================
# #%%
#B=np.array([6,12])
h0=np.array([1])#,1.1])
Nrunder=100
# #Q=np.array([1e-2])
 
x0=0
x1=1
dx0=0.01
dx=dx0
nx=int((x1-x0)/dx)
#dt0 =np.array([0.1,0.01,0.003])
dt0 =np.array([1e-7])
t_max=1e4
Q=np.array([0.75])#,1e-3,1e-4,1e-5,1e-6])
hcut=np.array([0.30])#,0.4,0.3])
profilinterval=11
#%%
# 
np.savez('simparametere.npz',h0=h0,dt0=dt0,dx=dx,Nrunder=Nrunder,x0=x0,x1=x1,nx=nx,t_max=t_max,Q=Q,hcut=hcut)  # save simulation set parameters to file

#%% Kjøre
ih=0
for iQ in range(len(Q)):
    for ih in range(len(hcut)):
        
#for it in range(2,3):
#    for iB in range(17,21):
        cwd=cwd0
        Qstr=str(Q[iQ])
        hstr=str(hcut[ih])
        #hstr=str(h0[ih])          
#             
        mappe='Q='+Qstr+', hcut='+hstr              
        path = os.path.join(cwd,mappe)
        if not os.path.exists(path):
            os.mkdir(path)    
        os.chdir(path)
        cwd = os.getcwd()
        tick=0
        for n in range(Nrunder):
            mappe='sim_'+str(n)         
            path = os.path.join(cwd,mappe)
            if not os.path.exists(path):
                os.mkdir(path)    
                os.chdir(path)
                if tick==0:
                    tick=fxn.flukt_int_TF( Q[iQ],h0[0] , x0, x1, nx, dt0,t_max,1,profilinterval,n,hcut[ih])
                else:
                    tick=fxn.flukt_int_TF( Q[iQ],h0[0] , x0, x1, nx, dt0,100,1,profilinterval,n,hcut[ih])
                #np.savez('simdata.npz',tR=tR,hmin=hmin)
            os.chdir(cwd)    
os.chdir(cwd0)

