#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 13:28:52 2023

@author: vira
"""

#exit()
#%reset -f
import os
import importlib
filedir=os.path.dirname(os.path.abspath(__file__))
os.chdir(filedir)

import synapsefunksjoner as fxn
importlib.reload(fxn)#%%#%%
#%%

from fenics import *
#from mshr import *
import numpy as np
import sys
import matplotlib.pyplot as plt
import time
import datetime
import csv
from decimal import Decimal
from scipy.signal import argrelextrema
#%%


folder='prote1n/Begge/TEST3_litengamma' 
path1 = os.path.join(os.getcwd(),folder)
if not os.path.exists(path1):
    os.mkdir(path1)
os.chdir(folder)
cwd0 = os.getcwd()

profilyn=1

Q=np.array([0.01])
K=1
gamma=0.2#0.5
h0=np.array([1.25])

tau1off=1e-3

sigmaon=0.2

kB=1.38e-23

l2=45e-9
l1=15e-9
L=20e-6
lscale=2e-7/1

fact=1
deltax=1*fact
dt = 3
x0=0
x1=(1*L/lscale)*fact
nx=int((x1-x0)/deltax)
t_max=1000
profilinterval1=3
profilinterval2=30


#sigmaoff=float(sigmaoff)

Nrunder=1




#%%
np.savez('simparametere.npz',Q=Q,K=K,gamma=gamma,h0=h0,deltax=deltax,dt=dt,t_max=t_max,Nrunder=Nrunder,L=L,x1=x1,nx=nx,sigmaon=sigmaon,tau1off=tau1off)  # save simulation set parameters to file


#%% Kjøre

for iQ in range(len(Q)):
    for ih in range(len(h0)):
        cwd=cwd0
        Qstr=str(Q[iQ])
        h0str=str(h0[ih]) 
        mappe='Q='+Qstr+', h0='+h0str 
        path = os.path.join(cwd,mappe)
        if not os.path.exists(path):
            os.mkdir(path)    
        profilyn=1
        os.chdir(path)
        cwd = os.getcwd()
        for n in range(Nrunder):
            mappe='sim_'+str(n)         
            path = os.path.join(cwd,mappe)
            if n==3:
                profilyn=0
            if not os.path.exists(path):
                os.mkdir(path)    
                os.chdir(path)
                fxn.fasesep_prote1n_begge(Q[iQ],K,gamma,h0[ih],l2,tau1off,sigmaon,x1,nx,profilinterval,profilinterval2,profilyn,deltax,dt,t_max,n)
            os.chdir(cwd) 
os.chdir(cwd0)


