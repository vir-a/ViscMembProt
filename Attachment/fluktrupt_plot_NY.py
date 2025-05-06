#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 11:30:35 2023

@author: vira
"""

%reset -f

#%%
import os
import numpy as np
import matplotlib.pyplot as plt


#%%
filedir=os.path.dirname(os.path.abspath(__file__))
os.chdir(filedir)
#%%

folder='FIN' 
intormemb=2

path1 = os.path.join(os.getcwd(),folder)
os.chdir(folder)
cwd0 = os.getcwd()
#%% last inn data hvis simuleringer allerede gjort

# =============================tRs================================================
resultarrays=np.load('data.npz')


tRs=resultarrays['tRs']
tRerrors=resultarrays['tRerrors']
logtRs=resultarrays['logtRs']
logtRerrors=resultarrays['logtRerrors']

#%%
inputdatas=np.load('simparametere.npz')
#deltax=inputdatas['deltax']
dt0=inputdatas['dt0']
#tau1off=inputdatas['tau1off']
#sigmaon=inputdatas['sigmaon']
Nrunder=inputdatas['Nrunder']
#L=inputdatas['L']
x1=inputdatas['x1']
nx=inputdatas['nx']
t_max=inputdatas['t_max']
Q=inputdatas['Q']
h0=inputdatas['h0']
hcut=inputdatas['hcut']
iQ=0
#exponents=np.array([11,14,15,8])

cols=['r','b','g','y','c','k','coral','navy','slategrey'] 
syms=['s','o','d','^','v','*','p','P','X']
linestyles=[(0, (4, 1)),(0, (6, 1, 1, 1, 1, 1)),(0, (8, 1)),(0, (3, 1, 1, 1)),'solid']


#%% Bindingstid vs h0 for forskjellige Q

# =============================================================================
# 
# fig2=plt.figure()#figsize=[4, 3.25])
# ax=plt.axes()
# ax.set_xscale('linear')
# ax.set_yscale('linear')
# #hran=np.array([h0[0],h0[-1]])
# 
# for iQ in range(len(Q)):
#     ax.errorbar(1-hcut,tRs[iQ,:],yerr=tRerrors[iQ,:],linestyle ='',marker='s',markersize=4,color=cols[iQ], label='Q = '+str(Q[iQ]))
# #for iQ in range(len(Qtrue)): 
# #    ax.errorbar(B,tRs[iQ,:],yerr=tRs_std[iQ,:],linestyle ='',marker='s',markersize=4,color=cols[iQ], label='Q = '+str(Qtrue[iQ]))
#     #ax.plot(Q, -4*h0[ih]**5*(np.log(Q/150)),cols[ih]+'--', label='$\sim -4h_0^5*ln(Q/150)$')
# #ax.set_xlim(0,2.9)
# #ax.set_ylim(0,30000)
#     #ax.plot(hran,np.exp(np.log(tRs[0,0])-exponents[iQ]*np.log(hran[0]))*hran**exponents[iQ],':',color=cols[0])
# #ax.plot(hran,0.5*tRs[1,0]*hran**10/(hran[0]**10),':',color=cols[1])
# ax.set_xlabel('$1-h^*$')
# ax.set_ylabel('Attachment time')
# 
# plt.legend(ncol=1,loc='upper left')
# plt.savefig('attachtime.eps',bbox_inches='tight')
# 
# =============================================================================

#%%


hran=np.linspace(hcut[0], hcut[-1],num=1000)
#hran=np.linspace(0, 2,num=1000)
rareevent=np.zeros((len(Q),len(hran)))
rareevent2=np.zeros((len(Q),len(hran)))
for iQ in range(len(Q)):
    if intormemb == 1:
        C=1e-6
        #powerlaw=tRs[iQ,0]*(hran/hran[0])**4
        rareevent=C*np.exp((6*(1-hran)**2)/((Q[0]**2)/2))
        rareevent2=C*np.exp((6*(1-hran)**2)/((Q[0]**2)/2))

    elif intormemb == 2:
        
        #rareevent=(tRs[iQ,0]/np.exp((60*(1-hcut[0])**2)/(4**2/2)))*np.exp((60*(1-hran)**2)/(4**2/2))
        #rareevent2=(tRs[iQ,0]/np.exp((360*(1-hcut[0])**2)/(4**2/2)))*np.exp((360*(1-hran)**2)/(4**2/2))
        #rareevent=C*np.exp(60*((1-hran)**2)/((Q[iQ]**2)/2))
        #rareevent2=C*np.exp(360*((1-hran)**2)/((Q[iQ]**2)/2))
        rareevent2[iQ,:]=(1/((1-hran)*((2*np.pi)**6)*((1/2)+(3/8)*(1-hran)**3)))*np.sqrt(4*((Q[iQ]**2)/2)*np.pi/((2*np.pi)**8))*np.exp(360*((1-hran)**2)/((Q[iQ]**2)/2))
        rareevent[iQ,:]=(1/((1-hran)*((2*np.pi)**6)*((1/2)+(3/8)*(1-hran)**3)))*np.sqrt(4*((Q[iQ]**2)/2)*np.pi/((2*np.pi)**8))*np.exp((4*np.pi**4)*((1-hran)**2)/((Q[iQ]**2)/2))
# =============================================================================
# #%%
# fig3=plt.figure()#figsize=[4, 3.25])
# ax=plt.axes()
# ax.set_xscale('log')
# ax.set_yscale('log')
# exponents=np.array([16,10.7,7.6])
# starts=np.array([2,3,6])
# for iQ in range(len(Q)):
#     ax.errorbar((1-hcut),tRs[iQ,:],yerr=tRerrors[iQ,:],linestyle ='',marker='s',markersize=4,color=cols[iQ], label='Q = '+str(Q[iQ]))
# #for iQ in range(len(Qtrue)): 
# #    ax.errorbar(B,tRs[iQ,:],yerr=tRs_std[iQ,:],linestyle ='',marker='s',markersize=4,color=cols[iQ], label='Q = '+str(Qtrue[iQ]))
#     #ax.plot(Q, -4*h0[ih]**5*(np.log(Q/150)),cols[ih]+'--', label='$\sim -4h_0^5*ln(Q/150)$')
#     ax.plot(1-hran,rareevent[iQ,:],color=cols[iQ],linestyle ='--',label='Q='+str(Q[iQ])+'_Cosine')
#     ax.plot(1-hran,rareevent2[iQ,:],color=cols[iQ],linestyle =':',label='Q='+str(Q[iQ])+'_Polynomial profile + cosine prefactor')
#     #ax.plot(hran,powerlaw)
# 
# ax.set_ylim(np.nanmin(tRs),np.nanmax(tRs))
#     #ax.plot(hran,np.exp(np.log(tRs[iQ,starts[iQ]])-exponents[iQ]*np.log(h0[starts[iQ]]))*hran**exponents[iQ],':',color=cols[iQ], label='α = '+str(exponents[iQ]))
# #ax.plot(hran,0.5*tRs[1,0]*hran**10/(hran[0]**10),':',color=cols[1])
# #ax.set_ylim(0.75*np.nanmin(tRs),1.5*np.nanmax(tRs))
# #ax.set_xlim(0.9*np.min(h0),1.1*np.max(h0))
# ax.set_xlabel('$1-h^*$')
# ax.set_ylabel('Attachment time')
# 
# plt.legend(ncol=2,loc='upper left')
# plt.savefig('attachtime_log.eps',bbox_inches='tight')
# 
# 
# =============================================================================
# =============================================================================
# #%%
# fig3=plt.figure()#figsize=[4, 3.25])
# ax=plt.axes()
# ax.set_xscale('linear')
# ax.set_yscale('log')
# 
# for iQ in range(len(Q)):
#     ax.errorbar(1-hcut,tRs[iQ,:],yerr=tRerrors[iQ,:],linestyle ='',marker='s',markersize=4,color=cols[iQ], label='Q = '+str(Q[iQ]))
# #for iQ in range(len(Qtrue)): 
# #    ax.errorbar(B,tRs[iQ,:],yerr=tRs_std[iQ,:],linestyle ='',marker='s',markersize=4,color=cols[iQ], label='Q = '+str(Qtrue[iQ]))
#     #ax.plot(Q, -4*h0[ih]**5*(np.log(Q/150)),cols[ih]+'--', label='$\sim -4h_0^5*ln(Q/150)$')
# 
#     ax.plot(1-hran,rareevent[iQ,:],color=cols[iQ],linestyle ='--',label='Q='+str(Q[iQ])+'_Cosine')
#     ax.plot(1-hran,rareevent2[iQ,:],color=cols[iQ],linestyle =':',label='Q='+str(Q[iQ])+'_Polynomial profile + cosine prefactor')
#    #ax.plot(hran,powerlaw)
#  #   ax.plot(hran,np.exp(np.log(tRs[iQ,2])-exponents[iQ]*np.log(h0[2]))*hran**exponents[iQ],':',color=cols[iQ])
# #ax.plot(hran,0.5*tRs[1,0]*hran**10/(hran[0]**10),':',color=cols[1])
# #ax.set_ylim(0.75*np.nanmin(tRs),1.5*np.nanmax(tRs))
# #ax.set_xlim(0.9*np.min(h0),1.1*np.max(h0))
# ax.set_xlabel('$1-h^*$')
# ax.set_ylabel('Attachment time')
# ax.set_ylim(np.nanmin(tRs),np.nanmax(tRs))
# 
# plt.legend(ncol=1,loc='upper left')
# plt.savefig('attachtime_semilogy.eps',bbox_inches='tight')
# 
# 
# 
# =============================================================================
#%%
# =============================================================================
# fig3=plt.figure()#figsize=[4, 3.25])
# ax=plt.axes()
# ax.set_xscale('log')
# ax.set_yscale('log')
# 
# for iQ in range(len(Q)):
#     #ax.errorbar(1-hcut,logtRs[iQ,:]-np.log(C),yerr=logtRerrors,linestyle ='',marker='s',markersize=4,color=cols[iQ], label='Q = '+str(Q[iQ]))
#     ax.errorbar(1-hcut,np.log(tRs[iQ,:]/C),yerr=tRerrors[iQ,:]/(tRs[iQ,:]),linestyle ='',marker='s',markersize=4,color=cols[iQ], label='Q = '+str(Q[iQ]))
# 
# #for iQ in range(len(Qtrue)): 
# #    ax.errorbar(B,tRs[iQ,:],yerr=tRs_std[iQ,:],linestyle ='',marker='s',markersize=4,color=cols[iQ], label='Q = '+str(Qtrue[iQ]))
#     #ax.plot(Q, -4*h0[ih]**5*(np.log(Q/150)),cols[ih]+'--', label='$\sim -4h_0^5*ln(Q/150)$')
# 
# #ax.set_ylim(45,60)
#    # ax.plot(1-hran,np.log(rareevent/C),label='Sharp')
#     ax.plot(1-hran,np.log(rareevent2/C),label='Round')
#     #ax.plot(hran,powerlaw)
#  #   ax.plot(hran,np.exp(np.log(tRs[iQ,2])-exponents[iQ]*np.log(h0[2]))*hran**exponents[iQ],':',color=cols[iQ])
# #ax.plot(hran,0.5*tRs[1,0]*hran**10/(hran[0]**10),':',color=cols[1])
# #ax.set_ylim(0.75*np.nanmin(tRs),1.5*np.nanmax(tRs))
# #ax.set_xlim(0.9*np.min(h0),1.1*np.max(h0))
# ax.set_xlabel('$1-h^*$')
# ax.set_ylabel('log($t_R$/C)')
# 
# plt.legend(ncol=1,loc='lower right')
# plt.savefig('attachtime_test.eps',bbox_inches='tight')
# 
# =============================================================================
#%%
# =============================================================================
# fig3=plt.figure(figsize=[4, 3.25])
# ax=plt.axes()
# ax.set_xscale('linear')
# ax.set_yscale('log')
# 
# for iQ in range(len(Q)):
#     #ax.errorbar(1-hcut,logtRs[iQ,:]-np.log(C),yerr=logtRerrors,linestyle ='',marker='s',markersize=4,color=cols[iQ], label='Q = '+str(Q[iQ]))
#     ax.errorbar((1-hcut)**2,tRs[iQ,:],yerr=tRerrors[iQ,:],linestyle ='',marker='s',markersize=4,color=cols[iQ], label='Q = '+str(Q[iQ]))
# 
# #for iQ in range(len(Qtrue)): 
# #    ax.errorbar(B,tRs[iQ,:],yerr=tRs_std[iQ,:],linestyle ='',marker='s',markersize=4,color=cols[iQ], label='Q = '+str(Qtrue[iQ]))
#     #ax.plot(Q, -4*h0[ih]**5*(np.log(Q/150)),cols[ih]+'--', label='$\sim -4h_0^5*ln(Q/150)$')
# 
# #ax.set_ylim(45,60)
#     ax.plot((1-hran)**2,rareevent[iQ,:],color=cols[iQ],linestyle ='--',label='Q='+str(Q[iQ])+'_Cosine')
#     ax.plot((1-hran)**2,rareevent2[iQ,:],color=cols[iQ],linestyle =':',label='Q='+str(Q[iQ])+'_Polynomial profile + cosine prefactor')
#   #ax.plot(hran,powerlaw)
#  #   ax.plot(hran,np.exp(np.log(tRs[iQ,2])-exponents[iQ]*np.log(h0[2]))*hran**exponents[iQ],':',color=cols[iQ])
# #ax.plot(hran,0.5*tRs[1,0]*hran**10/(hran[0]**10),':',color=cols[1])
# #ax.set_ylim(0.75*np.nanmin(tRs),1.5*np.nanmax(tRs))
# #ax.set_xlim(0.9*np.min(h0),1.1*np.max(h0))
# ax.set_xlabel('$(h_0-h^*)^2$')
# ax.set_ylabel('Attachment time')
# 
# plt.legend(ncol=1,loc='upper left')
# plt.savefig('attachtime_str8.eps',bbox_inches='tight')
# 
# =============================================================================
#%%
fig3=plt.figure(figsize=[4, 3.25])
ax=plt.axes()
ax.set_xscale('linear')
ax.set_yscale('log')

for iQ in range(len(Q)):
    ax.plot((1-hran)**2,rareevent2[iQ,:],color=cols[iQ],linestyle='-')
    #ax.errorbar(1-hcut,logtRs[iQ,:]-np.log(C),yerr=logtRerrors,linestyle ='',marker='s',markersize=4,color=cols[iQ], label='Q = '+str(Q[iQ]))
    ax.errorbar((1-hcut)**2,tRs[iQ,:],yerr=tRerrors[iQ,:],linestyle ='',marker=syms[iQ],markersize=4,color=cols[iQ], label='$Q='+str(Q[iQ])+'$')

#for iQ in range(len(Qtrue)): 
#    ax.errorbar(B,tRs[iQ,:],yerr=tRs_std[iQ,:],linestyle ='',marker='s',markersize=4,color=cols[iQ], label='Q = '+str(Qtrue[iQ]))
    #ax.plot(Q, -4*h0[ih]**5*(np.log(Q/150)),cols[ih]+'--', label='$\sim -4h_0^5*ln(Q/150)$')

#ax.set_ylim(45,60)
    #ax.plot((1-hran)**2,rareevent,label='Cosine')
    
    #ax.plot(hran,powerlaw)
 #   ax.plot(hran,np.exp(np.log(tRs[iQ,2])-exponents[iQ]*np.log(h0[2]))*hran**exponents[iQ],':',color=cols[iQ])
#ax.plot(hran,0.5*tRs[1,0]*hran**10/(hran[0]**10),':',color=cols[1])
#ax.set_ylim(0.75*np.nanmin(tRs),1.5*np.nanmax(tRs))
#ax.set_xlim(0.9*np.min(h0),1.1*np.max(h0))
ax.set_xlabel('$(h_0-h^*)^2$')
ax.set_ylabel(r'$\langle t_B \rangle $')
ax.set_ylim(np.nanmin(tRs)*0.75,np.nanmax(tRs)*1.5)

plt.legend(ncol=1,loc='lower right')
plt.savefig('attachtime_Nice.eps',bbox_inches='tight')
