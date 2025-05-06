#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:49:26 2024

@author: vira
"""
import os

from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import csv
from decimal import Decimal
from scipy.signal import argrelextrema
#%%
#%%
def profiler(datapoints):
    t=np.genfromtxt('t_profil.txt') # read txt file with times
    Nt=len(t)
    Np=datapoints  # number of timepoints to sample
    
    ## equally spaced in solution
    #n1=int(Nt/Np)
    #ints=linspace(1,Nt-1,Np)  # create vector of times to sample


    ## equally spaced in time
    tmax=t[-1]
    ts=np.linspace(t[0],t[-1],Np)
    #print(t)
    ints=np.zeros(Np)
    for i in range(Np):
        ti=ts[i]
        diff=abs(t-ti)
        ints[i]=np.argmin(diff)
 
    ints=ints.astype(int)

    ts=t[ints]            # record the times that will be sampled
    intss=ints+1
    print('Timepoints:')
    print(ts)
    with open('profiler.csv') as fil:    # Open csv with profiles and read out x axis
        reader=csv.reader(fil)
        x=[row for idx, row in enumerate(reader) if idx==0]
    x=np.array(x[0]) # make into 1-D array
    x=x[:-1]      # remove commas at the end
    x=x.astype(float) # string to float
    #print(x)

    #print(x)
    with open('profiler.csv') as hfil: # Open csv with profiles and h profiles
        hreader=csv.reader(hfil)
        h=[hrow for idxx, hrow in enumerate(hreader) if idxx in intss] # select timepoints
    #print(h)
    h=np.array(h) # list to array
    h=h[:,:-1] # remove commas
    h=h.astype(float)  # string to float
    #print(h)
    #for i in range(len(ints)):
    #    plt.plot(x,h[i,:])
    #plt.show()

    np.savez('profildata.npz', h=h,x=x,ts=ts)  # save data to file


def profilplot():
    
    
    
    # load data and assign to appropriate arrays
    data=np.load('profildata.npz')
    x=data['x']
    ts=data['ts']
    h=data['h']
    #print(ts)
    #print(h)

    cols=['g','b','r','k','y']
    # plot raw profiles
    plt.figure(figsize=[4, 3.25])
    for i in range(len(ts)):
        plt.plot(x,h[i,:], label='t = '+str(round(ts[i],7)),color=cols[i])
    #    plt.plot(x,h[i,:], label='t = '+"{:.2E}".format(Decimal(str(ts[i]))))

    plt.xlabel('x')
    plt.ylabel('h')
    plt.legend(loc='upper left')
    plt.ylim(0,2*max(h[0,:]))#1.1*max([max(h[0,:]),max(h[-1,:])]))

    plt.savefig('rawprofiles.png')
    plt.close()
    

def gjennomsnittsprofil(N,h0,hcut,intormemb,x1):
    cwd = os.getcwd()
    for n in range(N):
        mappe='sim_'+str(n)         
        path = os.path.join(cwd,mappe)
        os.chdir(path)
    
        # load data and assign to appropriate arrays
        data=np.load('profildata.npz')
        x=data['x']
        x=x/x1
        nx=len(x)
        imid=nx//2
        ts=data['ts']
        h=data['h']
        h=h[-1,:]
        min =1000
        if n ==0:
            hs=np.zeros((N,nx))
        for ix in range(nx):
            val=h[ix]
            if val < min:
                min=val
                imin=ix
        diff=imid-imin
        hs[n,:]=np.roll(h,diff)
        os.chdir(cwd) 

    xfinleft=np.linspace(x[0],x[-1]/2,1000)
    xfinright=np.linspace(x[-1]/2,x[-1],1000)
    
    
    
    hintleft=6*h0*(-xfinleft**2+1/4)
    hintright=6*h0*(-xfinright**2+2*xfinright-3/4)
    

    
    cols=['g','b','r','k','y']
    # plot raw profiles
    plt.figure(figsize=[4, 3.25])#
    hmean=np.mean(hs, axis=0)
    plt.plot(x,hmean,markersize=3, marker='o', color='black',linestyle='none',label='Mean')
    if intormemb == 1:
        
        hintleft=hcut+6*(h0-hcut)*(-xfinleft**2+1/4)
        hintright=hcut+6*(h0-hcut)*(-xfinright**2+2*xfinright-3/4)
        plt.plot(xfinleft,hintleft,'y-')
        plt.plot(xfinright,hintright,'y-')
    elif intormemb == 2:
        h0temp = np.mean(hmean)
        print(h0temp)
        #hcos=h0temp+(h0temp-hcut)*np.cos(2*np.pi*x)
        #hmembright=hcut+(h0-hcut)*5*((xfinright-0.5)**4-2*(xfinright-0.5)**3+(xfinright-0.5))
        hmembleft_alt=hcut+30*(h0-hcut)*((xfinleft+0.5)**4-2*(xfinleft+0.5)**3+(xfinleft+0.5)**2)
        hmembright_alt=hcut+30*(h0-hcut)*((xfinright-0.5)**4-2*(xfinright-0.5)**3+(xfinright-0.5)**2)
        #plt.plot(x,hcos,'y-',label='Cosine')
        #plt.plot(xfinleft,hmembleft,'y-')
        plt.plot(xfinright,hmembright_alt,'b-',label='Theory')
        plt.plot(xfinleft,hmembleft_alt,'b-')
        
        
# =============================================================================
#         hintleft=6*h0*(-xfinleft**2+1/4)
#         hintright=6*h0*(-xfinright**2+2*xfinright-3/4)
#         plt.plot(xfinleft,hintleft,'g:')
#         plt.plot(xfinright,hintright,'g:')
# =============================================================================
        #plt.plot(xfinright,hmembright_alt,'g:')
        #plt.plot(xfinleft,hmembleft_alt,'g:')
    plt.xlabel('x')
    plt.ylabel('h')
    plt.legend(loc='lower left')
    #plt.ylim(0,2*max(h[0,:]))#1.1*max([max(h[0,:]),max(h[-1,:])]))
    xran=np.array([x[0],x[-1]])
    hcrit=np.array([hcut,hcut])
    plt.plot(xran,hcrit,'k:')
    plt.savefig('FIN_meanprofile.eps',bbox_inches='tight')
    #plt.close()
    
    
def profildetalj():
    ts=np.genfromtxt('t_profil.txt') # read txt file with times
    Nt=len(ts)
  # number of timepoints to sample


    with open('profiler.csv') as fil:    # Open csv with profiles and read out x axis
        reader=csv.reader(fil)
        x=[row for idx, row in enumerate(reader) if idx==0]
    x=np.array(x[0]) # make into 1-D array
    x=x[:-1]      # remove commas at the end
    x=x.astype(float) # string to float

    with open('profiler.csv') as hfil: # Open csv with profiles and h profiles
        hreader=csv.reader(hfil)
        h=[hrow for idxx, hrow in enumerate(hreader) if idxx != 0 ] # select timepoints
    #print(h)
    h=h[:-1]
    h=np.array(h) # list to array
    h=h[:,:-1] # remove commas
    h=h.astype(float)  # string to float
    #print(h)
    #for i in range(len(ints)):
    #    plt.plot(x,h[i,:])
    #plt.show()

    np.savez('profildetaljer.npz', h=h,x=x,ts=ts)  # save data to file


def flukt_int_TF( Q, h0 , x0, x1, nx, dt0, t_end, profilyn,profilinterval,simnum,hcut):
    #profilyn: 1 = skriv profiler, 0 = ikke
    Q=float(Q)
    
    h0=float(h0)
    dt0=float(dt0)
    starttime=time.time()
    # Define mesh
    dt=Constant(dt0)
    deltax=(x1-x0)/nx
    #Q=Q/(sqrt(deltax)*sqrt(dt0))
    mesh = IntervalMesh(nx,x0,x1)
    class Periodic_sides(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0) and on_boundary

        def map(self, x, y):
            y[0] = x[0] - x1
            
    # Define mixed function space
    S = FiniteElement('P',interval,1)
    element = MixedElement([S,S])
    VW = FunctionSpace(mesh,element, constrained_domain=Periodic_sides())
    # Define initial condition 
    hinf=h0
    hp_0 = Expression((' hinf','0'), degree=1,hinf=hinf)
    # Define trial and test functions within function space
    v, w = TestFunctions(VW)
    hp = Function(VW)
    h, p = split(hp)
    # Initialize solution in our function space at t=0
    hp_n=project(hp_0, VW)
    h_n, p_n = split(hp_n)
    # Make eta a function
    elmts = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    etaV=FunctionSpace(mesh,elmts)#, constrained_domain=Periodic_sides())
    eta=Function(etaV)
    # Prepare to start 
    t=0
    #xdmffile = XDMFFile('solution.xdmf') #create xdmf file for the solution
    print('t = '+str(round(t,4)))
    # Regn ut opprinelig minimumspunkt
    nxd=nx
    xd=np.linspace(x0,x1,num=nxd)
    min=1000
    max=1000
    xmin=0
    for i in range(nxd-1):
        val=hp_n(xd[i])[0]
        if val < min:
            min=val
            idm=i
    xmin=xd[idm]
    print('old_height = '+str(round(min,5)))
    ## Formulate the variational problem
    # Write out weak form of the equations with all terms on one side
    F=h*v*dx-h_n*v*dx-dt*(1/2)*Q*v.dx(0)*(h**(3/2)+h_n**(3/2))*eta*dx-dt*(1/2)*(h**3+h_n**3)*dot(grad(p),grad(v))*dx +p*w*dx+dot(grad(w),grad(h))*dx
    J=derivative(F,hp)
    problem=NonlinearVariationalProblem(F,hp,J=J)
    solver=NonlinearVariationalSolver(problem)
    # lag filer der data kan lagres
    h_series=open('h.txt','w')
    t_series=open('t.txt','w')   # create and open text files in which to write down parameter values
    #rough_series=open('rough.txt','w')
    t_profiles=open('t_profil.txt','w')
    fil=open('profiler.csv','w') #csv file with film profiles at each timestep
    x=mesh.coordinates()
    h_series.write(str(round(min,6)))       # write value
    h_series.write('\n')
                      #new line
    t_series.write(str(round(t,6)))
    t_series.write('\n')
    def stochastic(Number,deltax,dt):
        mesh = UnitIntervalMesh(Number)        # lag mesh
        elms = FiniteElement('Lagrange', mesh.ufl_cell(), 1)  #
        U = FunctionSpace(mesh, elms)     # funksjonsrom,
        u = TrialFunction(U)
        v = TestFunction(U)
        eta = Function(U)
        
        stdevinv = np.sqrt(deltax)*np.sqrt(float(dt))
        stdev=1/stdevinv
        def update_random(random=eta):    # funksjon
            random.vector().set_local(np.random.normal(0, stdev, random.vector().local_size()))
            random.vector().apply('insert')
        eq = inner(u, v)*dx - inner(eta, v)*dx
        u = Function(U)
        update_random()
        solve(lhs(eq) == rhs(eq), u)
        outfile = open('u.txt', 'w')
        for i in range(len(mesh.coordinates())):
            outfile.write('%f\t %f\n' %(mesh.coordinates()[i], u(mesh.coordinates()[i])))
        outfile.close()
    if profilyn==1:
        t_profiles.write(str(round(t,6)))
        t_profiles.write('\n')
        for n in range(len(x)):
            fil.write(str(x[n,0])+',')
        fil.write('\n')
        for n in range(len(x)):
            fil.write(str(hp_n(x[n,0])[0])+',')
        fil.write('\n')
    else:
        for n in range(len(x)):
            fil.write(str(x[n,0])+',')
        fil.write('\n')
        for n in range(len(x)):
            fil.write(str(0)+',')
        fil.write('\n')
        for n in range(len(x)):
            fil.write(str(0.6)+',')
        fil.write('\n')
        for n in range(len(x)):
            fil.write(str(0.4)+',')
        fil.write('\n')
        for n in range(len(x)):
            fil.write(str(0.4)+',')
        fil.write('\n')
        for n in range(len(x)):
            fil.write(str(0.4)+',')
        fil.write('\n')
        t_profiles.write(str(round(0,6)))
        t_profiles.write('\n')
        t_profiles.write(str(round(1,6)))
        t_profiles.write('\n')
        t_profiles.write(str(round(2,6)))
        t_profiles.write('\n')
        t_profiles.write(str(round(2,6)))
        t_profiles.write('\n')
        t_profiles.write(str(round(2,6)))
        t_profiles.write('\n')
    # registrer metadata
    simpars=open('simparametere.txt','w')
    simpars.write('nx = '+str(nx))
    simpars.write('\n')
    simpars.write('x0 = '+str(x0))
    simpars.write('\n')
    simpars.write('x1 = '+str(x1))
    simpars.write('\n')
    simpars.write('dt_0 = '+str(dt0))
    simpars.write('\n')
    #simpars.write('lambda (wavelength) = '+str(lambd))
   # simpars.write('\n')
    simpars.write('Q (Fluctuation strength) = '+str(Q))
    simpars.write('\n')

    simpars.write('Initial film height = '+str(hinf))
    simpars.write('\n')
    # Initialisér løsning
    F0=h*v*dx-h_n*v*dx-dt*(dot(grad(p),grad(v))-dot(grad(h),grad(v)))*dx +p*w*dx+dot(grad(w),grad(h))*dx
    solve(F0==0,hp)
    # Loop i tid
    h_old =10
    hmax_old=12
    itno_old =0
    i=0
    imax=-1000
    roughnesses=open('roughnesses.txt','w')
    while min>hcut and t<t_end:
        i+=1
        print('\n   \n')
        print( 'Q = '+str(Q)+', h0 = '+str(h0)+' -- Sim nummer '+str(simnum))
        # update current timestep
        t+=float(dt)                       # Time for which we will be computing solution
        print('Iteration number: '+str(i))#+'  /'+str(Nt))
        print('dt = '+str(float(dt)))
        print('t = '+str(round(float(t),7)))#+'  /'+str(T))
        print('old_min_height = '+str(min))
        print('old_max_height = '+str(max))
        stochastic(nx,deltax,dt)    # oppdater eta i tekstfil
        etaraw=np.loadtxt('u.txt') # last inn eta
        etaraw=etaraw[:,1]          # ta bare andre kolonne
        for ik in range(len(etaraw)):
            eta.vector()[ik] = etaraw[ik]         # sett vektorverdiene inn i funksjonen eta                         *********************
            if (np.sqrt(eta.vector()[ik]**2-etaraw[ik]**2)) != 0:
                print('Fail to assign new values to eta')
        # solve the variational problem to get new u at current timestep
        itno,conv =solver.solve()
        _h,_p = hp.split()
        min=1000
        max=-1000
        for ix in range(nxd-1):
            val=hp_n(xd[ix])[0]
            if val < min:
                min=val
                idm=ix
            if val>max:
                max=val
                imax=ix
        #xmin=xd[idm]
        #xmax=xd[imax]
        print('new_min height = '+str(min))
        print('new_max height = '+str(max))
        #print('min position = '+str(round(xmin,5)))
        #print('max position = '+str(round(xmax,5)))
        h_series.write(str(min))
        h_series.write('\n')
        t_series.write(str(t))
        t_series.write('\n')
        #L_series.write(str(L))
        #L_series.write('\n')
        #print('new_length = '+str(round(L,5)))
        if profilyn==1 and i/profilinterval==int(i/profilinterval):
            t_profiles.write(str(t))
            t_profiles.write('\n')
            for n in range(len(x)):
                fil.write(str(_h(x[n,0]))+',')
            fil.write('\n')
        #xdmffile.write(_h,t)
        # Make the current timestep solution the previous timestep for the next iteration
        hp_n.assign(hp)
        # Adapt timestep size
        deltah=min-h_old
        deltahmax=max-hmax_old
        #print('deltahmin= '+ str(deltah))
        #print('deltahmax= '+ str(deltahmax))
        h_old=min
        hmax_old=max
# =============================================================================
#         profilfil=open('profil.csv','w') #csv file with film profiles at each timestep
#         for n in range(len(x)):
#             profilfil.write(str(_h(x[n,0]))+',')
#         profilfil.close()
#         with open('profil.csv') as hfil: # Open csv with profiles and h profiles
#             hreader=csv.reader(hfil)
#             hruf=[hrow for idxx, hrow in enumerate(hreader) if idxx == 0 ] # select timepoints
#         #print(h)
#         hruf=np.array(hruf) # list to array
#  
#         hruf=hruf[:,:-1] # remove commas
#         hruf=hruf.astype(float)  # string to float
#         #print(hruf)
#         rough=np.sqrt(np.sum((hruf[:]-h0)**2)/nx)
#         roughnesses.write(str(round(rough,6)))
#         roughnesses.write('\n')
#         #print('Roughness = ')
#         #print(str(rough))
# =============================================================================
        
        _dt=dt
        dt.assign(_dt)
        h_old=min
        itno_old =itno
        
    tick=0
    
    t_profiles.write(str(round(t,12)))
    t_profiles.write('\n')
    if t>0.99*t_end:
        t=np.nan
        tick=1    
    roughnesses.close()
    h_series.close()
    t_series.close()

    for n in range(len(x)):
        fil.write(str(_h(x[n,0]))+',')
    fil.write('\n')
    fil.close()
    t_profiles.close()

    endtime=time.time()
    print('Elapsed time = '+str(datetime.timedelta(seconds=endtime-starttime)))
    simpars.write('Elapsed real time = '+str(datetime.timedelta(seconds=endtime-starttime)))
    simpars.write('\n')
    simpars.write('# of iterations = '+str(i))
    simpars.write('\n')
    simpars.write('Simulation end time = '+str(t))
    simpars.close()
    print('Rupture time = '+str(t))
    np.savez('rupturetime.npz',tR=t)
    return tick


def flukt_memb_TF( Q, h0 , x0, x1, nx, dt0, t_end, profilyn,profilinterval,simnum,hcut):
    #profilyn: 1 = skriv profiler, 0 = ikke
    Q=float(Q)
    h0=float(h0)
    dt0=float(dt0)
    starttime=time.time()
    # Define mesh
    dt=Constant(dt0)
    deltax=(x1-x0)/nx
    mesh = IntervalMesh(nx,x0,x1)
    class Periodic_sides(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0) and on_boundary

        def map(self, x, y):
            y[0] = x[0] - x1
            
    # Define mixed function space
    EL = FiniteElement('P',interval,1)
    element = MixedElement([EL,EL,EL])
    VWS = FunctionSpace(mesh,element, constrained_domain=Periodic_sides())
    # Define initial condition 
    hinf=h0
    hpm_0 = Expression((' hinf','0','0'), degree=1,hinf=hinf)
    # Define trial and test functions within function space
    v, w,s = TestFunctions(VWS)
    hpm = Function(VWS)
    h,p,m, = split(hpm)
    # Initialize solution in our function space at t=0
    hpm_n=project(hpm_0, VWS)
    h_n, p_n,m_n = split(hpm_n)
    # Make eta a function
    elmts = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    etaV=FunctionSpace(mesh,elmts)#, constrained_domain=Periodic_sides())
    eta=Function(etaV)
    # Prepare to start 
    t=0
    #xdmffile = XDMFFile('solution.xdmf') #create xdmf file for the solution
    print('t = '+str(round(t,4)))
    # Regn ut opprinelig minimumspunkt
    nxd=nx
    xd=np.linspace(x0,x1,num=nxd)
    min=1000
    max=1000
    xmin=0
    for i in range(nxd-1):
        val=hpm_n(xd[i])[0]
        if val < min:
            min=val
            idm=i
    xmin=xd[idm]
    print('old_height = '+str(round(min,5)))
    ## Formulate the variational problem
    # Write out weak form of the equations with all terms on one side
    F=h_n*v*dx -h*v*dx-dt*Q*h**(3/2)*v.dx(0)*eta*dx-dt*h**3*dot(grad(p),grad(v))*dx +p*w*dx +dot(grad(w),grad(m))*dx +m*s*dx +dot(grad(s),grad(h))*dx 
    J=derivative(F,hpm)
    problem=NonlinearVariationalProblem(F,hpm,J=J)
    solver=NonlinearVariationalSolver(problem)
    # lag filer der data kan lagres
    h_series=open('h.txt','w')
    t_series=open('t.txt','w')   # create and open text files in which to write down parameter values
    #rough_series=open('rough.txt','w')
    t_profiles=open('t_profil.txt','w')
    fil=open('profiler.csv','w') #csv file with film profiles at each timestep
    x=mesh.coordinates()
    h_series.write(str(round(min,6)))       # write value
    h_series.write('\n')
                      #new line
    t_series.write(str(round(t,6)))
    t_series.write('\n')
    def stochastic(Number,deltax,dt):
        mesh = UnitIntervalMesh(Number)        # lag mesh
        elms = FiniteElement('Lagrange', mesh.ufl_cell(), 1)  #
        U = FunctionSpace(mesh, elms)     # funksjonsrom,
        u = TrialFunction(U)
        v = TestFunction(U)
        eta = Function(U)
        
        stdevinv = np.sqrt(deltax)*np.sqrt(float(dt))
        stdev=1/stdevinv
        def update_random(random=eta):    # funksjon
            random.vector().set_local(np.random.normal(0, stdev, random.vector().local_size()))
            random.vector().apply('insert')
        eq = inner(u, v)*dx - inner(eta, v)*dx
        u = Function(U)
        update_random()
        solve(lhs(eq) == rhs(eq), u)
        outfile = open('u.txt', 'w')
        for i in range(len(mesh.coordinates())):
            outfile.write('%f\t %f\n' %(mesh.coordinates()[i], u(mesh.coordinates()[i])))
        outfile.close()
    if profilyn==1:
        t_profiles.write(str(round(t,6)))
        t_profiles.write('\n')
        for n in range(len(x)):
            fil.write(str(x[n,0])+',')
        fil.write('\n')
        for n in range(len(x)):
            fil.write(str(hpm_n(x[n,0])[0])+',')
        fil.write('\n')
    else:
        for n in range(len(x)):
            fil.write(str(x[n,0])+',')
        fil.write('\n')
        for n in range(len(x)):
            fil.write(str(0)+',')
        fil.write('\n')
        for n in range(len(x)):
            fil.write(str(0.6)+',')
        fil.write('\n')
        for n in range(len(x)):
            fil.write(str(0.4)+',')
        fil.write('\n')
        for n in range(len(x)):
            fil.write(str(0.4)+',')
        fil.write('\n')
        for n in range(len(x)):
            fil.write(str(0.4)+',')
        fil.write('\n')
        t_profiles.write(str(round(0,6)))
        t_profiles.write('\n')
        t_profiles.write(str(round(1,6)))
        t_profiles.write('\n')
        t_profiles.write(str(round(2,6)))
        t_profiles.write('\n')
        t_profiles.write(str(round(2,6)))
        t_profiles.write('\n')
        t_profiles.write(str(round(2,6)))
        t_profiles.write('\n')
    # registrer metadata
    simpars=open('simparametere.txt','w')
    simpars.write('nx = '+str(nx))
    simpars.write('\n')
    simpars.write('x0 = '+str(x0))
    simpars.write('\n')
    simpars.write('x1 = '+str(x1))
    simpars.write('\n')
    simpars.write('dt_0 = '+str(dt0))
    simpars.write('\n')
    #simpars.write('lambda (wavelength) = '+str(lambd))
   # simpars.write('\n')
    simpars.write('Q (Fluctuation strength) = '+str(Q))
    simpars.write('\n')

    simpars.write('Initial film height = '+str(hinf))
    simpars.write('\n')
    # Initialisér løsning
    F0=h*v*dx-h_n*v*dx +p*w*dx-p_n*w*dx   +m*s*dx -m_n*s*dx 
    solve(F0==0,hpm)
    # Loop i tid
    h_old =10
    hmax_old=12
    itno_old =0
    i=0
    imax=-1000
    roughnesses=open('roughnesses.txt','w')
    while min>hcut and t<t_end:
        i+=1
        print('\n   \n')
        print( 'Q = '+str(Q)+', h0 = '+str(h0)+' -- Sim nummer '+str(simnum))
        # update current timestep
        t+=float(dt)                       # Time for which we will be computing solution
        print('Iteration number: '+str(i))#+'  /'+str(Nt))
        print('dt = '+str(float(dt)))
        print('t = '+str(round(float(t),7)))#+'  /'+str(T))
        print('old_min_height = '+str(min))
        print('old_max_height = '+str(max))
        stochastic(nx,deltax,dt)    # oppdater eta i tekstfil
        etaraw=np.loadtxt('u.txt') # last inn eta
        etaraw=etaraw[:,1]          # ta bare andre kolonne
        for ik in range(len(etaraw)):
            eta.vector()[ik] = etaraw[ik]         # sett vektorverdiene inn i funksjonen eta                         *********************
            if (np.sqrt(eta.vector()[ik]**2-etaraw[ik]**2)) != 0:
                print('Fail to assign new values to eta')
        # solve the variational problem to get new u at current timestep
        itno,conv =solver.solve()
        _h,_p,_m = hpm.split()
        min=1000
        max=-1000
        for ix in range(nxd-1):
            val=hpm_n(xd[ix])[0]
            if val < min:
                min=val
                idm=ix
            if val>max:
                max=val
                imax=ix
        #xmin=xd[idm]
        #xmax=xd[imax]
        print('new_min height = '+str(min))
        print('new_max height = '+str(max))
        #print('min position = '+str(round(xmin,5)))
        #print('max position = '+str(round(xmax,5)))
        h_series.write(str(min))
        h_series.write('\n')
        t_series.write(str(t))
        t_series.write('\n')
        #L_series.write(str(L))
        #L_series.write('\n')
        #print('new_length = '+str(round(L,5)))
        if profilyn==1 and i/profilinterval==int(i/profilinterval):
            t_profiles.write(str(t))
            t_profiles.write('\n')
            for n in range(len(x)):
                fil.write(str(_h(x[n,0]))+',')
            fil.write('\n')
        #xdmffile.write(_h,t)
        # Make the current timestep solution the previous timestep for the next iteration
        hpm_n.assign(hpm)
        # Adapt timestep size
        deltah=min-h_old
        deltahmax=max-hmax_old
        #print('deltahmin= '+ str(deltah))
        #print('deltahmax= '+ str(deltahmax))
        h_old=min
        hmax_old=max
# =============================================================================
#         profilfil=open('profil.csv','w') #csv file with film profiles at each timestep
#         for n in range(len(x)):
#             profilfil.write(str(_h(x[n,0]))+',')
#         profilfil.close()
#         with open('profil.csv') as hfil: # Open csv with profiles and h profiles
#             hreader=csv.reader(hfil)
#             hruf=[hrow for idxx, hrow in enumerate(hreader) if idxx == 0 ] # select timepoints
#         #print(h)
#         hruf=np.array(hruf) # list to array
#  
#         hruf=hruf[:,:-1] # remove commas
#         hruf=hruf.astype(float)  # string to float
#         #print(hruf)
#         rough=np.sqrt(np.sum((hruf[:]-h0)**2)/nx)
#         roughnesses.write(str(round(rough,6)))
#         roughnesses.write('\n')
#         #print('Roughness = ')
#         #print(str(rough))
# =============================================================================
        
        _dt=dt
        dt.assign(_dt)
        h_old=min
        itno_old =itno
        
    tick=0
    t_profiles.write(str(round(t,12)))
    t_profiles.write('\n')
    if t>0.99*t_end:
        t=np.nan
        tick=1    
    roughnesses.close()
    h_series.close()
    t_series.close()
    
    for n in range(len(x)):
        fil.write(str(_h(x[n,0]))+',')
    fil.write('\n')
    fil.close()
    t_profiles.close()

    endtime=time.time()
    print('Elapsed time = '+str(datetime.timedelta(seconds=endtime-starttime)))
    simpars.write('Elapsed real time = '+str(datetime.timedelta(seconds=endtime-starttime)))
    simpars.write('\n')
    simpars.write('# of iterations = '+str(i))
    simpars.write('\n')
    simpars.write('Simulation end time = '+str(t))
    simpars.close()
    print('Rupture time = '+str(t))
    np.savez('rupturetime.npz',tR=t)
    return tick

def flukt_memb_TF_crank( Q, h0 , x0, x1, nx, dt0, t_end, profilyn,profilinterval,simnum,hcut):
    #profilyn: 1 = skriv profiler, 0 = ikke
    Q=float(Q)
    
    h0=float(h0)
    dt0=float(dt0)
    
    starttime=time.time()
    # Define mesh
    dt=Constant(dt0)
    deltax=(x1-x0)/nx
    #Q=Q/(np.sqrt(deltax)*np.sqrt(dt0))
    mesh = IntervalMesh(nx,x0,x1)
    class Periodic_sides(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0) and on_boundary

        def map(self, x, y):
            y[0] = x[0] - x1
            
    # Define mixed function space
    EL = FiniteElement('P',interval,1)
    element = MixedElement([EL,EL,EL])
    VWS = FunctionSpace(mesh,element, constrained_domain=Periodic_sides())
    # Define initial condition 
    hinf=h0
    hpm_0 = Expression((' hinf','0','0'), degree=1,hinf=hinf)
    # Define trial and test functions within function space
    v, w,s = TestFunctions(VWS)
    hpm = Function(VWS)
    h,p,m, = split(hpm)
    # Initialize solution in our function space at t=0
    hpm_n=project(hpm_0, VWS)
    h_n, p_n,m_n = split(hpm_n)
    # Make eta a function
    elmts = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    etaV=FunctionSpace(mesh,elmts)#, constrained_domain=Periodic_sides())
    eta=Function(etaV)
    # Prepare to start 
    t=0
    #xdmffile = XDMFFile('solution.xdmf') #create xdmf file for the solution
    print('t = '+str(round(t,4)))
    # Regn ut opprinelig minimumspunkt
    nxd=nx
    xd=np.linspace(x0,x1,num=nxd)
    min=1000
    max=1000
    xmin=0
    for i in range(nxd-1):
        val=hpm_n(xd[i])[0]
        if val < min:
            min=val
            idm=i
    xmin=xd[idm]
    print('old_height = '+str(round(min,5)))
    ## Formulate the variational problem
    # Write out weak form of the equations with all terms on one side
    F=h_n*v*dx -h*v*dx-dt*(1/2)*Q*(h**(3/2)+h_n**(3/2))*v.dx(0)*eta*dx-dt*(1/2)*(h**3+h_n**3)*dot(grad(p),grad(v))*dx +p*w*dx +dot(grad(w),grad(m))*dx +m*s*dx +dot(grad(s),grad(h))*dx 
    J=derivative(F,hpm)
    problem=NonlinearVariationalProblem(F,hpm,J=J)
    solver=NonlinearVariationalSolver(problem)
    # lag filer der data kan lagres
    h_series=open('h.txt','w')
    t_series=open('t.txt','w')   # create and open text files in which to write down parameter values
    #rough_series=open('rough.txt','w')
    t_profiles=open('t_profil.txt','w')
    fil=open('profiler.csv','w') #csv file with film profiles at each timestep
    x=mesh.coordinates()
    h_series.write(str(round(min,6)))       # write value
    h_series.write('\n')
                      #new line
    t_series.write(str(round(t,6)))
    t_series.write('\n')
    def stochastic(Number,deltax,dt):
        mesh = UnitIntervalMesh(Number)        # lag mesh
        elms = FiniteElement('Lagrange', mesh.ufl_cell(), 1)  #
        U = FunctionSpace(mesh, elms)     # funksjonsrom,
        u = TrialFunction(U)
        v = TestFunction(U)
        eta = Function(U)
        
        stdevinv = np.sqrt(deltax)*np.sqrt(float(dt))
        stdev=1/stdevinv
        def update_random(random=eta):    # funksjon
            random.vector().set_local(np.random.normal(0, stdev, random.vector().local_size()))
            random.vector().apply('insert')
        eq = inner(u, v)*dx - inner(eta, v)*dx
        u = Function(U)
        update_random()
        solve(lhs(eq) == rhs(eq), u)
        outfile = open('u.txt', 'w')
        for i in range(len(mesh.coordinates())):
            outfile.write('%f\t %f\n' %(mesh.coordinates()[i], u(mesh.coordinates()[i])))
        outfile.close()
    if profilyn==1:
        t_profiles.write(str(round(t,6)))
        t_profiles.write('\n')
        for n in range(len(x)):
            fil.write(str(x[n,0])+',')
        fil.write('\n')
        for n in range(len(x)):
            fil.write(str(hpm_n(x[n,0])[0])+',')
        fil.write('\n')
    else:
        for n in range(len(x)):
            fil.write(str(x[n,0])+',')
        fil.write('\n')
        for n in range(len(x)):
            fil.write(str(0)+',')
        fil.write('\n')
        for n in range(len(x)):
            fil.write(str(0.6)+',')
        fil.write('\n')
        for n in range(len(x)):
            fil.write(str(0.4)+',')
        fil.write('\n')
        for n in range(len(x)):
            fil.write(str(0.4)+',')
        fil.write('\n')
        for n in range(len(x)):
            fil.write(str(0.4)+',')
        fil.write('\n')
        t_profiles.write(str(round(0,6)))
        t_profiles.write('\n')
        t_profiles.write(str(round(1,6)))
        t_profiles.write('\n')
        t_profiles.write(str(round(2,6)))
        t_profiles.write('\n')
        t_profiles.write(str(round(2,6)))
        t_profiles.write('\n')
        t_profiles.write(str(round(2,6)))
        t_profiles.write('\n')
    # registrer metadata
    simpars=open('simparametere.txt','w')
    simpars.write('nx = '+str(nx))
    simpars.write('\n')
    simpars.write('x0 = '+str(x0))
    simpars.write('\n')
    simpars.write('x1 = '+str(x1))
    simpars.write('\n')
    simpars.write('dt_0 = '+str(dt0))
    simpars.write('\n')
    #simpars.write('lambda (wavelength) = '+str(lambd))
   # simpars.write('\n')
    simpars.write('Q (Fluctuation strength) = '+str(Q))
    simpars.write('\n')

    simpars.write('Initial film height = '+str(hinf))
    simpars.write('\n')
    # Initialisér løsning
    F0=h*v*dx-h_n*v*dx +p*w*dx-p_n*w*dx   +m*s*dx -m_n*s*dx 
    solve(F0==0,hpm)
    # Loop i tid
    h_old =10
    hmax_old=12
    itno_old =0
    i=0
    imax=-1000
    roughnesses=open('roughnesses.txt','w')
    while min>hcut and t<t_end:
        i+=1
        print('\n   \n')
        print( 'Q = '+str(Q)+', hcut = '+str(hcut)+' -- Sim nummer '+str(simnum))
        # update current timestep
        t+=float(dt)                       # Time for which we will be computing solution
        print('Iteration number: '+str(i))#+'  /'+str(Nt))
        print('dt = '+str(float(dt)))
        print('t = '+str(round(float(t),7)))#+'  /'+str(T))
        print('old_min_height = '+str(min))
        print('old_max_height = '+str(max))
        stochastic(nx,deltax,dt)    # oppdater eta i tekstfil
        etaraw=np.loadtxt('u.txt') # last inn eta
        etaraw=etaraw[:,1]          # ta bare andre kolonne
        for ik in range(len(etaraw)):
            eta.vector()[ik] = etaraw[ik]         # sett vektorverdiene inn i funksjonen eta                         *********************
            if (np.sqrt(eta.vector()[ik]**2-etaraw[ik]**2)) != 0:
                print('Fail to assign new values to eta')
        # solve the variational problem to get new u at current timestep
        itno,conv =solver.solve()
        _h,_p,_m = hpm.split()
        min=1000
        max=-1000
        for ix in range(nxd-1):
            val=hpm_n(xd[ix])[0]
            if val < min:
                min=val
                idm=ix
            if val>max:
                max=val
                imax=ix
        #xmin=xd[idm]
        #xmax=xd[imax]
        print('new_min height = '+str(min))
        print('new_max height = '+str(max))
        #print('min position = '+str(round(xmin,5)))
        #print('max position = '+str(round(xmax,5)))
        h_series.write(str(min))
        h_series.write('\n')
        t_series.write(str(t))
        t_series.write('\n')
        #L_series.write(str(L))
        #L_series.write('\n')
        #print('new_length = '+str(round(L,5)))
        if profilyn==1 and i/profilinterval==int(i/profilinterval):
            t_profiles.write(str(t))
            t_profiles.write('\n')
            for n in range(len(x)):
                fil.write(str(_h(x[n,0]))+',')
            fil.write('\n')
        #xdmffile.write(_h,t)
        # Make the current timestep solution the previous timestep for the next iteration
        hpm_n.assign(hpm)
        # Adapt timestep size
        deltah=min-h_old
        deltahmax=max-hmax_old
        #print('deltahmin= '+ str(deltah))
        #print('deltahmax= '+ str(deltahmax))
        h_old=min
        hmax_old=max
# =============================================================================
#         profilfil=open('profil.csv','w') #csv file with film profiles at each timestep
#         for n in range(len(x)):
#             profilfil.write(str(_h(x[n,0]))+',')
#         profilfil.close()
#         with open('profil.csv') as hfil: # Open csv with profiles and h profiles
#             hreader=csv.reader(hfil)
#             hruf=[hrow for idxx, hrow in enumerate(hreader) if idxx == 0 ] # select timepoints
#         #print(h)
#         hruf=np.array(hruf) # list to array
#  
#         hruf=hruf[:,:-1] # remove commas
#         hruf=hruf.astype(float)  # string to float
#         #print(hruf)
#         rough=np.sqrt(np.sum((hruf[:]-h0)**2)/nx)
#         roughnesses.write(str(round(rough,6)))
#         roughnesses.write('\n')
#         #print('Roughness = ')
#         #print(str(rough))
# =============================================================================
        
        _dt=dt
        dt.assign(_dt)
        h_old=min
        itno_old =itno
        
    tick=0
    t_profiles.write(str(round(t,12)))
    t_profiles.write('\n')
    if t>0.99*t_end:
        t=np.nan
        tick=1    
    roughnesses.close()
    h_series.close()
    t_series.close()
    
    for n in range(len(x)):
        fil.write(str(_h(x[n,0]))+',')
    fil.write('\n')
    fil.close()
    t_profiles.close()

    endtime=time.time()
    print('Elapsed time = '+str(datetime.timedelta(seconds=endtime-starttime)))
    simpars.write('Elapsed real time = '+str(datetime.timedelta(seconds=endtime-starttime)))
    simpars.write('\n')
    simpars.write('# of iterations = '+str(i))
    simpars.write('\n')
    simpars.write('Simulation end time = '+str(t))
    simpars.close()
    print('Rupture time = '+str(t))
    np.savez('rupturetime.npz',tR=t)
    return tick

def hvtplots(hlim):
    t=np.genfromtxt('t.txt')
    h=np.genfromtxt('h.txt')
    Nt=len(t)
    tb1=1111
    it=0
    while h[it]>hlim and t[it]<t[-1]:
        it+=1
        tb1=t[it]
        
    if tb1>=t[-1]:
        tb1=np.nan
        
    #dt=t[1]-t[0]
    #it0=round(t0/dt)
    #it1=round(t1/dt)
    #tf=round(20/dt)
    #plt.semilogy(t[t0:tf],1-h[t0:tf])
    #fit=np.polyfit(t[it0:it1], np.log(h0-h[it0:it1]), 1)
    #growthrate=fit[0]
    tran=np.array([t[0],t[-1]])
    hcrit=np.array([hlim,hlim])
    #plt.semilogy(t[2:], h0-h[2:],'ok',markersize=3)
    #plt.semilogy(t[it0:it1],(h0-h[it0])*np.exp(growthrate*t[it0:it1]),'--y',linewidth=1.5)
    fig1=plt.figure()
    plt.plot(t,h,'ok',markersize=3)
    plt.plot(tran,hcrit,'g:')
    plt.xlabel('$t$')
    plt.ylabel('$h$')
    plt.savefig('hvt.eps')
    plt.close()
    
    fig1=plt.figure()
    plt.loglog(t,h,'ok',markersize=3)
    plt.loglog(tran,hcrit,'g:')
    plt.xlabel('$t$')
    plt.ylabel('$h$')
    plt.savefig('hvt_loglog.eps')
    plt.close()
    
    
    fig1=plt.figure()
    plt.semilogx(t,h,'ok',markersize=3)
    plt.semilogx(tran,hcrit,'g:')
    plt.xlabel('$t$')
    plt.ylabel('$h$')
    plt.savefig('hvt_semilogx.eps')
    plt.close()
    
