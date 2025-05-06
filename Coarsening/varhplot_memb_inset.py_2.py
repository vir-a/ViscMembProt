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
# =============================================================================
# folder='prote1n/Interfacial/Nyrunde_varh'
# xtri1=1.6e4
# xtri2=5e4
# trisize1=2
# trisize2=2.5
# x2tri1=xtri1*trisize1
# x2tri2=xtri2*trisize2
# trifact1=4.6
# trifact2=18094
# 
# =============================================================================

# =============================================================================
# folder='prote1n/Interfacial_constantmob/varhtest'
# xtri1=3e4
# xtri2=1e4
# trisize1=2
# trisize2=2.5
# x2tri1=xtri1*trisize1
# x2tri2=xtri2*trisize2
# trifact1=3.7
# trifact2=6594
# 
# =============================================================================
# =============================================================================
# folder='prote1n/Bending/Comparemobilities/JUL/Viscous' 
# xtri1=8e3
# xtri2=3.9e4
# trisize1=2
# trisize2=2.5
# x2tri1=xtri1*trisize1
# x2tri2=xtri2*trisize2
# trifact1=2
# trifact2=1594
# 
# =============================================================================


# =============================================================================
# folder='prote1n/Bending/Comparemobilities/JUL/Constantmob' 
# xtri1=9e3
# xtri2=3e4
# trisize1=2
# trisize2=2.5
# x2tri1=xtri1*trisize1
# x2tri2=xtri2*trisize2
# trifact1=1.0
# trifact2=894
# =============================================================================


folder='prote1n/Bending/CompareMobilities/JUL/Viscous' 
xtri1=5.5e3
xtri2=2.3e4
trisize1=2.7
trisize2=3.6
x2tri1=xtri1*trisize1
x2tri2=xtri2*trisize2
trifact1=2
trifact2=1594


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



#fig1=plt.figure(figsize=[4, 3.25])
fig,ax1=plt.subplots(figsize=[4, 3.25])
left, bottom, width, height = [0.15, 0.60, 0.25, 0.25]
ax2 = fig.add_axes([left, bottom, width, height])
#ax=plt.axes()
tick=0
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
        
# =============================================================================
#         
#         if ih != 1 and ih != 3:
#             inputdatas=np.load('Sizefactorsvt_SO.npz')
#             Lc=inputdatas['L']
#             t=inputdatas['t']
#             ax1.plot(t,Lc,color=cols[ih],marker=syms[ih],label='$h_0$ = '+h0str)
# 
# =============================================================================
        inputdatas=np.load('Sizefactorsvt_SO.npz')
        Lc=inputdatas['L']
        t=inputdatas['t']
        #ax1.plot(t,Lc,color=cols[ih],marker=syms[ih],label='$h_0$ = '+h0str)
        ax1.plot(t,Lc,color=cols[ih],linestyle=linestyles[ih],label='$h_0$ = '+h0str)

ax1.plot([xtri1, x2tri1],[(xtri1*trifact1)**(1/3),(x2tri1*trifact1)**(1/3)],'k-')
ax1.plot([xtri1, x2tri1],[(xtri1*trifact1)**(1/3),((xtri1*trifact1))**(1/3)],'k-')
ax1.plot([x2tri1, x2tri1],[(xtri1*trifact1)**(1/3),(x2tri1*trifact1)**(1/3)],'k-')
ax1.text(x2tri1+820, (31/33)*np.sqrt((xtri1*trifact1)**(1/3) * (x2tri1*trifact1)**(1/3)), "1", va='bottom', ha='left', fontsize=10)
ax1.text((9/10)*np.sqrt(xtri1*x2tri1), (35/40)*(xtri1*trifact1)**(1/3), "3", va='bottom', ha='left', fontsize=10)

ax1.plot([xtri2, x2tri2],[(xtri2*trifact2)**(1/5),(x2tri2*trifact2)**(1/5)],'k-')
ax1.plot([xtri2, x2tri2],[(xtri2*trifact2)**(1/5),((xtri2*trifact2))**(1/5)],'k-')
ax1.plot([x2tri2, x2tri2],[(xtri2*trifact2)**(1/5),(x2tri2*trifact2)**(1/5)],'k-')
ax1.text(x2tri2*(33/31), (18/19)*np.sqrt((xtri2*trifact2)**(1/5) * (x2tri2*trifact2)**(1/5)), "1", va='bottom', ha='left', fontsize=10)
ax1.text((19/20)*np.sqrt(xtri2*x2tri2), (8/9)*(xtri2*trifact2)**(1/5), "5", va='bottom', ha='left', fontsize=10)


os.chdir(cwd0)

#ax.plot(profts,kchars,'rx',label='fft')
#ax.plot(ts,kchar1s,'bx',label='fft')
#ax.plot(profts[it0:it1],(exp(fit[1])/(1))*profts[it0:it1]**exponent,'--b',linewidth=1.5,label='$\propto t^{'+str(round(exponent,2))+'}$')
ax1.set_xlabel('$t$')
ax1.set_ylabel('$L$')
#ax.set_title('Q = '+str(Q))
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.legend(loc='lower right')







os.chdir(filedir)
folder='prote1n/Bending/Comparemobilities/JUL/Constantmob' 
xtri1=1.5e4
xtri2=2.1e3
trisize1=5
trisize2=6
x2tri1=xtri1*trisize1
x2tri2=xtri2*trisize2
trifact1=0.5
trifact2=60124

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
        
        
# =============================================================================
#         if ih != 1 and ih != 3:
#             inputdatas=np.load('Sizefactorsvt_SO.npz')
#             Lc=inputdatas['L']
#             t=inputdatas['t']
#             ax.plot(t,Lc,color=cols[ih],marker=syms[ih],label='$h_0$ = '+h0str)
# =============================================================================

        inputdatas=np.load('Sizefactorsvt_SO.npz')
        Lc=inputdatas['L']
        t=inputdatas['t']
        #ax2.plot(t[1:],Lc[1:],color=cols[ih],marker=syms[ih],markersize=3,label='$h_0$ = '+h0str)
        ax2.plot(t[1:],Lc[1:],color=cols[ih],linestyle=linestyles[ih],label='$h_0$ = '+h0str)

ax2.plot([xtri1, x2tri1],[(xtri1*trifact1)**(1/3),(x2tri1*trifact1)**(1/3)],'k-')
ax2.plot([xtri1, x2tri1],[(xtri1*trifact1)**(1/3),((xtri1*trifact1))**(1/3)],'k-')
ax2.plot([x2tri1, x2tri1],[(xtri1*trifact1)**(1/3),(x2tri1*trifact1)**(1/3)],'k-')
ax2.text(x2tri1+820, (26/33)*np.sqrt((xtri1*trifact1)**(1/3) * (x2tri1*trifact1)**(1/3)), "1", va='bottom', ha='left', fontsize=10)
ax2.text((17/20)*np.sqrt(xtri1*x2tri1), (14/20)*(xtri1*trifact1)**(1/3), "3", va='bottom', ha='left', fontsize=10)

ax2.set_yscale('log')
ax2.set_xscale('log')
#ax2.xaxis.set_major_formatter(plt.NullFormatter())
#ax2.yaxis.set_major_formatter(plt.NullFormatter())
#ax2.yticks([])
#ax2.axes.get_yaxis().set_yticklabels([])
ax2.set_xticklabels([])
ax2.axes.get_yaxis().set_visible(False)
#ax2.set_yticklabels([])
ax2.plot([xtri2, x2tri2],[(xtri2*trifact2)**(1/5),(x2tri2*trifact2)**(1/5)],'k-')
ax2.plot([xtri2, x2tri2],[(xtri2*trifact2)**(1/5),((xtri2*trifact2))**(1/5)],'k-')
ax2.plot([x2tri2, x2tri2],[(xtri2*trifact2)**(1/5),(x2tri2*trifact2)**(1/5)],'k-')
ax2.text(x2tri2*(32/31), (63/76)*np.sqrt((xtri2*trifact2)**(1/5) * (x2tri2*trifact2)**(1/5)), "1", va='bottom', ha='left', fontsize=10)
ax2.text((16/20)*np.sqrt(xtri2*x2tri2), (11/16)*(xtri2*trifact2)**(1/5), "5", va='bottom', ha='left', fontsize=10)


os.chdir(cwd0)

#ax.plot(profts,kchars,'rx',label='fft')
#ax.plot(ts,kchar1s,'bx',label='fft')
#ax.plot(profts[it0:it1],(exp(fit[1])/(1))*profts[it0:it1]**exponent,'--b',linewidth=1.5,label='$\propto t^{'+str(round(exponent,2))+'}$')
#ax2.set_xlabel('$t$')
#ax2.set_ylabel('$L$')
#ax.set_title('Q = '+str(Q))

#ax2.legend(loc='lower right')

os.chdir(filedir)
plt.savefig('memb_varh_inset_2.eps',bbox_inches='tight')
