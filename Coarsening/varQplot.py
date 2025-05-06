#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 11:30:35 2023

@author: vira
"""

%reset -f


import os
import numpy as np
import matplotlib.pyplot as plt


#%%
filedir=os.path.dirname(os.path.abspath(__file__))
os.chdir(filedir)
#%%

#folder='prote1n/Bending/Comparemobilities/JUL/Viscous' 

folder='prote1n/Bending/prote1n_varQ_storN' 
xtri1=2.6e3
xtri2=1.6e4
trisize1=2
trisize2=2.5
x2tri1=xtri1*trisize1
x2tri2=xtri2*trisize2
trifact1=3.6
trifact2=1594
# =============================================================================
# folder='prote1n/Interfacial/Nyrunde_varQ'
# xtri1=2e4
# xtri2=6e4
# trisize1=2
# trisize2=2.5
# x2tri1=xtri1*trisize1
# x2tri2=xtri2*trisize2
# trifact1=3.6
# trifact2=9594
# 
# =============================================================================

path1 = os.path.join(os.getcwd(),folder)
os.chdir(folder)
cwd0 = os.getcwd()
#%%
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

#hcut=1.1

cols=['r','b','g','y','c','k','m'] 
syms=['s','o','d','^','v','*','p']
linestyles=[(0, (4, 1)),(0, (6, 1, 1, 1, 1, 1)),(0, (8, 1)),(0, (3, 1, 1, 1)),'solid',':']



fig1=plt.figure(figsize=[4, 3.25])
ax=plt.axes()
for iQ in range(len(Q)):
    for ih in range(len(h0)):
        cwd=cwd0
        Qstr=str(Q[iQ])
        h0str=str(h0[ih]) 
        mappe='Q='+Qstr+', h0='+h0str 
        path = os.path.join(cwd,mappe)  
        print( 'Hente data -- '+mappe)
        os.chdir(path)
        cwd = os.getcwd()
        #if ih != 1:
        inputdatas=np.load('Sizefactorsvt_SO.npz')
        Lc=inputdatas['L']
        t=inputdatas['t']
        #ax.plot(t,Lc,color=cols[iQ],marker=syms[iQ],label='$Q$ = '+Qstr)
        ax.plot(t,Lc,color=cols[iQ],linestyle=linestyles[iQ],label='$Q$ = '+Qstr)
plt.plot([xtri1, x2tri1],[(xtri1*trifact1)**(1/3),(x2tri1*trifact1)**(1/3)],'k-')
plt.plot([xtri1, x2tri1],[(xtri1*trifact1)**(1/3),((xtri1*trifact1))**(1/3)],'k-')
plt.plot([x2tri1, x2tri1],[(xtri1*trifact1)**(1/3),(x2tri1*trifact1)**(1/3)],'k-')
plt.text(x2tri1*(33/31), (31/33)*np.sqrt((xtri1*trifact1)**(1/3) * (x2tri1*trifact1)**(1/3)), "1", va='bottom', ha='left', fontsize=10)
plt.text((11/12)*np.sqrt(xtri1*x2tri1), (7/8)*(xtri1*trifact1)**(1/3), "3", va='bottom', ha='left', fontsize=10)

plt.plot([xtri2, x2tri2],[(xtri2*trifact2)**(1/5),(x2tri2*trifact2)**(1/5)],'k-')
plt.plot([xtri2, x2tri2],[(xtri2*trifact2)**(1/5),((xtri2*trifact2))**(1/5)],'k-')
plt.plot([x2tri2, x2tri2],[(xtri2*trifact2)**(1/5),(x2tri2*trifact2)**(1/5)],'k-')
plt.text(x2tri2*(33/31), (18/19)*np.sqrt((xtri2*trifact2)**(1/5) * (x2tri2*trifact2)**(1/5)), "1", va='bottom', ha='left', fontsize=10)
plt.text((9/10)*np.sqrt(xtri2*x2tri2), (7/8)*(xtri2*trifact2)**(1/5), "5", va='bottom', ha='left', fontsize=10)



os.chdir(cwd0)

#ax.plot(profts,kchars,'rx',label='fft')
#ax.plot(ts,kchar1s,'bx',label='fft')
#ax.plot(profts[it0:it1],(exp(fit[1])/(1))*profts[it0:it1]**exponent,'--b',linewidth=1.5,label='$\propto t^{'+str(round(exponent,2))+'}$')
ax.set_xlabel('t')
ax.set_ylabel('L')
#ax.set_title('Q = '+str(Q))
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend(loc='lower right')
plt.savefig('SOgrowths.eps',bbox_inches='tight')
# =============================================================================
# #%% things vs Q
# fig1=plt.figure()#figsize=[4, 3.25])
# ax=plt.axes()
# ax.set_xscale('log')
# for ih in range(len(h0)): 
#     ax.errorbar(Q,sizes[:,ih],yerr=sizeerrors[:,ih],marker='s',markersize=4,linestyle ='')
# #for iQ in range(len(Qtrue)): 
# #    ax.errorbar(B,tRs[iQ,:],yerr=tRs_std[iQ,:],linestyle ='',marker='s',markersize=4,color=cols[iQ], label='Q = '+str(Qtrue[iQ]))
#     #ax.plot(Q, -4*h0[ih]**5*(np.log(Q/150)),cols[ih]+'--', label='$\sim -4h_0^5*ln(Q/150)$')
# #ax.set_xlim(0,2.9)
# #ax.set_ylim(45,60)
# ax.set_xlabel('$Q$')
# ax.set_ylabel('Size factor')
# 
# #plt.legend(ncol=1,loc='lower left')
# plt.savefig('sizefactorvQ.eps',bbox_inches='tight')
# 
# 
# fig1=plt.figure()#figsize=[4, 3.25])
# ax=plt.axes()
# ax.set_xscale('log')
# for ih in range(len(h0)): 
#  ax.errorbar(Q,exps[:,ih],yerr=experrors[:,ih],marker='s',markersize=4,linestyle ='')
# #for iQ in range(len(Qtrue)): 
# #    ax.errorbar(B,tRs[iQ,:],yerr=tRs_std[iQ,:],linestyle ='',marker='s',markersize=4,color=cols[iQ], label='Q = '+str(Qtrue[iQ]))
#     #ax.plot(Q, -4*h0[ih]**5*(np.log(Q/150)),cols[ih]+'--', label='$\sim -4h_0^5*ln(Q/150)$')
# #ax.set_xlim(0,2.9)
# ax.set_ylim(0,1)
# ax.set_xlabel('$Q$')
# ax.set_ylabel('Exponent')
# 
# #plt.legend(ncol=1,loc='lower left')
# plt.savefig('exponentvQ.eps',bbox_inches='tight')
# 
# fig1=plt.figure()#figsize=[4, 3.25])
# ax=plt.axes()
# ax.set_xscale('log')
# for ih in range(len(h0)): 
#  ax.errorbar(Q,exp2s[:,ih],yerr=exp2errors[:,ih],marker='s',markersize=4,linestyle ='')
# #for iQ in range(len(Qtrue)): 
# #    ax.errorbar(B,tRs[iQ,:],yerr=tRs_std[iQ,:],linestyle ='',marker='s',markersize=4,color=cols[iQ], label='Q = '+str(Qtrue[iQ]))
#     #ax.plot(Q, -4*h0[ih]**5*(np.log(Q/150)),cols[ih]+'--', label='$\sim -4h_0^5*ln(Q/150)$')
# #ax.set_xlim(0,2.9)
# ax.set_ylim(0,1)
# ax.set_xlabel('$Q$')
# ax.set_ylabel('Exponent')
# 
# #plt.legend(ncol=1,loc='lower left')
# plt.savefig('FFT_exp_vQ.eps',bbox_inches='tight')
# 
# =============================================================================
# =============================================================================
# #%%
# ig1=plt.figure()#figsize=[4, 3.25])
# ax=plt.axes()
# ax.set_xscale('log')
# 
# for ih in range(len(h0)): 
#  ax.plot(Q,expSavs[:,ih],marker='s',markersize=4,linestyle ='')
# #for iQ in range(len(Qtrue)): 
# #    ax.errorbar(B,tRs[iQ,:],yerr=tRs_std[iQ,:],linestyle ='',marker='s',markersize=4,color=cols[iQ], label='Q = '+str(Qtrue[iQ]))
#     #ax.plot(Q, -4*h0[ih]**5*(np.log(Q/150)),cols[ih]+'--', label='$\sim -4h_0^5*ln(Q/150)$')
# #ax.set_xlim(0,2.9)
# ax.set_ylim(0,1)
# ax.set_xlabel('$Q$')
# ax.set_ylabel('Exponent')
# 
# #plt.legend(ncol=1,loc='lower left')
# plt.savefig('expSav_vQ.eps',bbox_inches='tight')
# #%% things vs h0
# fig1=plt.figure()#figsize=[4, 3.25])
# ax=plt.axes()
# ax.set_xscale('log')
# for iQ in range(len(Q)): 
#     ax.errorbar(h0,sizes[iQ,:],yerr=sizeerrors[iQ,:],marker='s',markersize=4,linestyle ='')
# #for iQ in range(len(Qtrue)): 
# #    ax.errorbar(B,tRs[iQ,:],yerr=tRs_std[iQ,:],linestyle ='',marker='s',markersize=4,color=cols[iQ], label='Q = '+str(Qtrue[iQ]))
#     #ax.plot(Q, -4*h0[ih]**5*(np.log(Q/150)),cols[ih]+'--', label='$\sim -4h_0^5*ln(Q/150)$')
# #ax.set_xlim(0,2.9)
# #ax.set_ylim(45,60)
# ax.set_xlabel('$h_0$')
# ax.set_ylabel('Size factor')
# 
# #plt.legend(ncol=1,loc='lower left')
# plt.savefig('sizefactorvh0.eps',bbox_inches='tight')
# 
# 
# fig1=plt.figure()#figsize=[4, 3.25])
# ax=plt.axes()
# ax.set_xscale('log')
# for ih in range(len(h0)): 
#  ax.errorbar(h0,exps[iQ,:],yerr=experrors[iQ,:],marker='s',markersize=4,linestyle ='')
# #for iQ in range(len(Qtrue)): 
# #    ax.errorbar(B,tRs[iQ,:],yerr=tRs_std[iQ,:],linestyle ='',marker='s',markersize=4,color=cols[iQ], label='Q = '+str(Qtrue[iQ]))
#     #ax.plot(Q, -4*h0[ih]**5*(np.log(Q/150)),cols[ih]+'--', label='$\sim -4h_0^5*ln(Q/150)$')
# #ax.set_xlim(0,2.9)
# ax.set_ylim(0,1)
# ax.set_xlabel('$h_0$')
# ax.set_ylabel('Exponent')
# 
# #plt.legend(ncol=1,loc='lower left')
# plt.savefig('exponentvh0.eps',bbox_inches='tight')
# 
# 
# fig1=plt.figure()#figsize=[4, 3.25])
# ax=plt.axes()
# ax.set_xscale('log')
# for ih in range(len(h0)): 
#  ax.errorbar(h0,exp2s[iQ,:],yerr=exp2errors[iQ,:],marker='s',markersize=4,linestyle ='')
# #for iQ in range(len(Qtrue)): 
# #    ax.errorbar(B,tRs[iQ,:],yerr=tRs_std[iQ,:],linestyle ='',marker='s',markersize=4,color=cols[iQ], label='Q = '+str(Qtrue[iQ]))
#     #ax.plot(Q, -4*h0[ih]**5*(np.log(Q/150)),cols[ih]+'--', label='$\sim -4h_0^5*ln(Q/150)$')
# #ax.set_xlim(0,2.9)
# ax.set_ylim(0,1)
# ax.set_xlabel('$h_0$')
# ax.set_ylabel('Exponent')
# 
# #plt.legend(ncol=1,loc='lower left')
# plt.savefig('FFT_exp_vh0.eps',bbox_inches='tight')
# #%%
# fig1=plt.figure()#figsize=[4, 3.25])
# ax=plt.axes()
# ax.set_xscale('log')
# iQ=0
# for iQ in range(len(Q)): 
#  ax.plot(h0,expSavs[iQ,:],marker='s',markersize=4,linestyle ='')
# #for iQ in range(len(Qtrue)): 
# #    ax.errorbar(B,tRs[iQ,:],yerr=tRs_std[iQ,:],linestyle ='',marker='s',markersize=4,color=cols[iQ], label='Q = '+str(Qtrue[iQ]))
#     #ax.plot(Q, -4*h0[ih]**5*(np.log(Q/150)),cols[ih]+'--', label='$\sim -4h_0^5*ln(Q/150)$')
# #ax.set_xlim(0,2.9)
# ax.set_ylim(0,0.5)
# ax.set_xlabel('$h_0$')
# ax.set_ylabel('Exponent')
# 
# #plt.legend(ncol=1,loc='lower left')
# plt.savefig('expSav_vh0.eps',bbox_inches='tight')
# 
# #%% Bindingstid vs h0 for forskjellige Q
# fig3=plt.figure()#figsize=[4, 3.25])
# ax=plt.axes()
# ax.set_xscale('linear')
# ax.set_yscale('log')
# 
# for iQ in range(len(Q)):
#     ax.errorbar(h0,tb1s[iQ,:],yerr=tb1errors[iQ,:],linestyle ='',marker='s',markersize=4,color=cols[iQ], label='Q = '+str(Q[iQ]))
#     #ax.plot(hran-hcut,rareevent,label='Cosine')
#     #ax.plot(hran-hcut,rareevent2,label='Polynomial profile + cosine prefactor')
#     #ax.plot((hran-hcut)**2,rareevent_test)
# #for iQ in range(len(Qtrue)): 
# #    ax.errorbar(B,tRs[iQ,:],yerr=tRs_std[iQ,:],linestyle ='',marker='s',markersize=4,color=cols[iQ], label='Q = '+str(Qtrue[iQ]))
#     #ax.plot(Q, -4*h0[ih]**5*(np.log(Q/150)),cols[ih]+'--', label='$\sim -4h_0^5*ln(Q/150)$')
# #ax.set_xlim(0,2.9)
# #ax.set_ylim(45,60)
# #ax.plot(hran,0.025*tb1s[0,0]*hran**15/(hran[0]**10),':',color=cols[0])
# #ax.plot(hran,0.5*tb1s[1,0]*hran**10/(hran[0]**10),':',color=cols[1])
# ax.set_xlabel('$h_0$')
# ax.set_ylabel('Attachment time')
# 
# plt.legend(ncol=1,loc='upper left')
# plt.savefig('attachtime_log.eps',bbox_inches='tight')
# #%% Bindingstid vs h0 for forskjellige Q
# hcut=1.5
# 
# hran=np.linspace(h0[0], h0[-1],num=1000)
# #hran=np.linspace(0, 2,num=1000)
# for iQ in range(len(Q)):
#     
#     rareevent2=1e12*(1/((hran-hcut)*((2*np.pi)**6)*((1/2)*hran**3+(3/8)*hran*(hran-hcut)**2)))*np.sqrt(4*(((Q[iQ]*100)**2)/2)*np.pi/((2*np.pi)**8))*np.exp((180*(hran-hcut)**2)/(((Q[iQ]*100)**2)/2))
#     rareevent=(1/((hran-hcut)*((2*np.pi)**6)*((1/2)*hran**3+(3/8)*hran*(hran-hcut)**2)))*np.sqrt(4*((Q[iQ]**2)/2)*np.pi/((2*np.pi)**8))*np.exp((4*np.pi**4)*((hran-hcut)**2)/(((Q[iQ]*10)**2)/2))
#     rareevent_test=1e4*np.exp((180*(hran-hcut)**2)/(((Q[iQ]*100)**2)/2))
#     #rareevent_test2=1e3*np.exp((209*(hran-hcut)**2)/(((Q[iQ]*100)**2)/2))
# fig3=plt.figure()#figsize=[4, 3.25])
# ax=plt.axes()
# ax.set_xscale('linear')
# ax.set_yscale('log')
# 
# for iQ in range(len(Q)):
#     ax.errorbar((h0-hcut)**2,tb1s[iQ,:],yerr=tb1errors[iQ,:],linestyle ='',marker='s',markersize=4,color=cols[iQ], label='Q = '+str(Q[iQ]))
#     #ax.plot(hran-hcut,rareevent,label='Cosine')
#     ax.plot((hran-hcut)**2,rareevent_test,label='R-E fit')
#     #ax.plot((hran-hcut)**2,rareevent_test2)
# #for iQ in range(len(Qtrue)): 
# #    ax.errorbar(B,tRs[iQ,:],yerr=tRs_std[iQ,:],linestyle ='',marker='s',markersize=4,color=cols[iQ], label='Q = '+str(Qtrue[iQ]))
#     #ax.plot(Q, -4*h0[ih]**5*(np.log(Q/150)),cols[ih]+'--', label='$\sim -4h_0^5*ln(Q/150)$')
# ax.set_xlim(0,np.max((hran-hcut)**2))
# #ax.set_ylim(45,60)
# #ax.plot(hran,0.025*tb1s[0,0]*hran**15/(hran[0]**10),':',color=cols[0])
# #ax.plot(hran,0.5*tb1s[1,0]*hran**10/(hran[0]**10),':',color=cols[1])
# ax.set_xlabel('$(h_0-h^*)^2$')
# ax.set_ylabel('Attachment time')
# 
# plt.legend(ncol=1,loc='lower right')
# plt.savefig('attachtime_log_RAREEVENT.eps',bbox_inches='tight')
# #%%
# 
# fig2=plt.figure()#figsize=[4, 3.25])
# ax=plt.axes()
# ax.set_xscale('linear')
# ax.set_yscale('linear')
# #hran=np.array([h0[0],h0[-1]])
# #hran=np.linspace(h0[0], h0[-1],num=50)
# 
# for iQ in range(len(Q)):
#     ax.errorbar(h0-hcut,tb1s[iQ,:],yerr=tb1errors[iQ,:],linestyle ='',marker='s',markersize=4,color=cols[iQ], label='Q = '+str(Q[iQ]))
#     #ax.plot(hran-hcut,rareevent,label='Cosine')
#     #ax.plot(hran-hcut,rareevent2,label='Polynomial profile + cosine prefactor')
# #for iQ in range(len(Qtrue)): 
# #    ax.errorbar(B,tRs[iQ,:],yerr=tRs_std[iQ,:],linestyle ='',marker='s',markersize=4,color=cols[iQ], label='Q = '+str(Qtrue[iQ]))
#     #ax.plot(Q, -4*h0[ih]**5*(np.log(Q/150)),cols[ih]+'--', label='$\sim -4h_0^5*ln(Q/150)$')
# #ax.set_xlim(0,2.9)
# #ax.set_ylim(0,30000)
# #ax.plot(hran,0.025*tb1s[0,0]*hran**15/(hran[0]**10),':',color=cols[0])
# #ax.plot(hran,0.5*tb1s[1,0]*hran**10/(hran[0]**10),':',color=cols[1])
# ax.set_xlabel('$h_0$')
# ax.set_ylabel('Attachment time')
# 
# plt.legend(ncol=1,loc='upper left')
# plt.savefig('attachtime.eps',bbox_inches='tight')
# 
# 
# 
# 
# #%%
# 
# =============================================================================
