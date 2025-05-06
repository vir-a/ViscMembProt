#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 13:28:52 2023

@author: vira
"""

#exit()
%reset -f
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

hcut=1.8
thresh=hcut
#folder='prote1n/Bending_constantmob/varh' 
#folder='prote1n/Interfacial_constantmob/varhtest/'
#folder='prote1n/Interfacial/Nyrunde_varh/'

#folder='prote1n/Bending/CompareMobilities/JUL/Darcy'
folder='prote1n/Bending/prote1n_varQ_storN' 
#folder='prote1n/Bending/gammelt/prote1n_varh0_N5' 
path1 = os.path.join(os.getcwd(),folder)

os.chdir(folder)
cwd0 = os.getcwd()

inputdatas=np.load('simparametere.npz')
deltax=inputdatas['deltax']
dt=inputdatas['dt']
tau1off=inputdatas['tau1off']
sigmaon=inputdatas['sigmaon']
Nrunder=inputdatas['Nrunder']
L=inputdatas['L']
x1=inputdatas['x1']
nx=inputdatas['nx']
t_max=inputdatas['t_max']
Q=inputdatas['Q']
h0=inputdatas['h0']
#h0=h0[2:4]
#%%
t0=2e3
t1=float(t_max)

#%% postprosesser og lagre data i mapper
#Nrunder=15
exp2avs=np.zeros((len(Q),len(h0)))
for iQ in range(len(Q)):
    for ih in range(len(h0)):
        realorno=1
        cwd=cwd0
        Qstr=str(Q[iQ])
        h0str=str(h0[ih]) 
        mappe='Q='+Qstr+', h0='+h0str 
        path = os.path.join(cwd,mappe)  
        print( 'Postprossesér -- '+mappe)
        os.chdir(path)
        cwd = os.getcwd()
        
        tb1=0
        exp=0
        sluttfaktor=0
        exp2=0
        slutt2=0
        charkavs=0
        expSav=0
        sizefact_average=0
        exp_sizefact_average=0
        print('begynner å gå inn i mapper for å regne ')
        for n in range(Nrunder):
            
            mappe='sim_'+str(n) 
            print( 'Simulering # '+str(n) )
            path = os.path.join(cwd,mappe)
            os.chdir(path)
            file_path = os.path.join(path, 'profiles.npz')
            if os.path.exists(file_path):
                
                profilfil=np.load('profiles.npz')
                xs=profilfil['xs']
                ys=profilfil['ys']
                profts=profilfil['profts']
                profiles=profilfil['profiles']
                if profts[-1]==0:
                    tick=0
                    ind=3
                    while tick<1:
                        if profts[ind]>0:
                            ind+=1
                        else:
                            tick=11
                    profts=profts[0:ind-1]
                    profiles=profiles[:,:,0:ind-1]
                #fxn.kontur(xs,ys,profts,profiles)
                
#                
## Gammel sizefactorberegning
                

                #sizefactors,areas, edges=fxn.flekkemåling(xs,ys,profts,profiles,deltax,Q[iQ],hcut)
                #np.savez('sizefactors.npz',areas=areas, edges=edges,sizefactors=sizefactors,profts=profts)
                #exp=fxn.growthrate(t0,t1,Q[iQ])
                #print('Old size factor calculation finished')
                #sluttfaktor=sizefactors[-1]
#
          #utdatert beregning av S for individuell sim     #charks=fxn.kcalc_indS(xs,ys,profts,profiles,deltax,Q[iQ],hcut)

                #np.savez('charks.npz',charks=charks,profts=profts)
                #tb1=fxn.bindingstid(1.0)
                
                #Growthrate -utdatert#exp2=fxn.NEWgrowthrate(t0,t1,Q[iQ])

                
            else:
                realorno=0
                exp=np.nan
                sluttfaktor=np.nan
        
            np.savez('verdier.npz',hcut=hcut,sluttfaktor=sluttfaktor,exp=exp,tb1=tb1,exp2=exp2,slutt2=slutt2)
            os.chdir(cwd)  
        print(realorno)
        if realorno>0.5:
            print('realtest passed')
            #  Slow ShinozakiOono method
            #charkavs,expSav=fxn.kcalc_Sav(xs,ys,profts,profiles,Nrunder,t0,t1)
            #  Counting points
            #effrads_average,exp_effrad_average=fxn.flekkemåling_gjsnitt(xs,ys,profts,profiles,deltax,hcut,Nrunder,t0,t1)
            #np.savez('Averaged_effrad.npz',effrads_average=effrads_average,exp_effrad_average=exp_effrad_average)
            #  Fast ShinozakiOono method
            charkavs_JBL,expSav_JBL=fxn.kcalc_JBL(xs,ys,profts,profiles,Nrunder,t0,t1)
            #exp2avs[iQ,ih]=expSav_JBL
            #np.savez('charkavs.npz',charkavs=charkavs,charkavs_JBL=charkavs_JBL)
            #  Ndomains stuff
            Ndomains,Ndomexp=fxn.Ndomains(xs,ys,profts,thresh,t0,t1,Nrunder)
            np.savez('Ndomains.npz',Ndomains=Ndomains,Ndomexp=Ndomexp)
            fxn.shellavg(xs,ys,profts,profiles,Nrunder,t0,t1)
            #charkavs_nequals2,expSav_nequals2=fxn.nequals2_(xs,ys,profts,profiles,Nrunder,t0,t1)
os.chdir(cwd0)
#np.savez('expSav.npz',exp2avs=exp2avs)
#%% aggreger relevante verdier for plotting
# =============================================================================
# 
# sizes=np.zeros((len(Q),len(h0)))
# sizeerrors=np.zeros((len(Q),len(h0)))
# tb1s=np.zeros((len(Q),len(h0)))
# tb1errors=np.zeros((len(Q),len(h0)))
# exps=np.zeros((len(Q),len(h0)))
# experrors=np.zeros((len(Q),len(h0)))
# exp2s=np.zeros((len(Q),len(h0)))
# exp2errors=np.zeros((len(Q),len(h0)))
# #exp2avs=np.zeros((len(Q),len(h0)))
# #exp2averrors=np.zeros((len(Q),len(h0)))
# for iQ in range(len(Q)):
#     for ih in range(len(h0)):
#         cwd=cwd0
#         Qstr=str(Q[iQ])
#         h0str=str(h0[ih]) 
#         mappe='Q='+Qstr+', h0='+h0str 
#         path = os.path.join(cwd,mappe)  
#         os.chdir(path)
#         cwd = os.getcwd()
#         
#         finalsizedist=np.zeros((Nrunder))
#         tb1dist=np.zeros((Nrunder))
#         expdist=np.zeros((Nrunder))
#         exp2dist=np.zeros((Nrunder))
#         for n in range(Nrunder):
#             mappe='sim_'+str(n)         
#             path = os.path.join(cwd,mappe)
#             os.chdir(path)
#             dataene=np.load('verdier.npz')
#             sluttfaktor=dataene['sluttfaktor']
#             exp=dataene['exp']
#             exp2=dataene['exp2']
#             #exp2av=dataene['exp2av']
#             tb1=dataene['tb1']
#             finalsizedist[n]=sluttfaktor
#             tb1dist[n]=tb1
#             expdist[n]=exp
#             exp2dist[n]=exp2
#             #exp2avdist[n]=exp2av
#             os.chdir(cwd)         
#         sizemean=np.mean(finalsizedist)
#         sizestd=np.std(finalsizedist)
#         sizes[iQ,ih]=sizemean
#         sizeerrors[iQ,ih]=sizestd
#         tb1mean=np.mean(tb1dist)
#         tb1std=np.std(tb1dist)
#         tb1s[iQ,ih]=tb1mean
#         tb1errors[iQ,ih]=tb1std
#         expmean=np.mean(expdist)
#         expstd=np.std(expdist)
#         exps[iQ,ih]=expmean
#         experrors[iQ,ih]=expstd
#         exp2mean=np.mean(exp2dist)
#         exp2std=np.std(exp2dist)
#         exp2s[iQ,ih]=exp2mean
#         exp2errors[iQ,ih]=exp2std
#         np.savez('dists',Qstr=Qstr,h0str=h0str,sizemean=sizemean,tb1mean=tb1mean,expmean=expmean,expstd=expstd,sizestd=sizestd,tb1dist=tb1dist,expdist=expdist,exp2dist=exp2dist,finalsizedist=finalsizedist)
# os.chdir(cwd0)
# np.savez('data.npz',sizes=sizes,sizeerrors=sizeerrors,tb1s=tb1s,tb1errors=tb1errors,exps=exps,experrors=experrors,exp2s=exp2s,exp2errors=exp2errors,hcut=hcut)  # save data to file
# 
# 
# =============================================================================

