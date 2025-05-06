#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 13:30:52 2023

@author: vira
"""
#from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import csv
from decimal import Decimal
from scipy.signal import argrelextrema
from fenics import *
import os
import datetime


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
    ts=np.linspace(t[1],t[-1],Np)
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


    # plot raw profiles
    plt.figure(figsize=[4, 3.25])
    for i in range(len(ts)):
        plt.plot(x,h[i,:], label='t = '+str(round(ts[i],7)))
    #    plt.plot(x,h[i,:], label='t = '+"{:.2E}".format(Decimal(str(ts[i]))))

    plt.xlabel('x')
    plt.ylabel('h')
    plt.legend(loc='upper left')
    #plt.ylim(0,1.4*max(h[:,:]))#1.1*max([max(h[0,:]),max(h[-1,:])]))

    plt.savefig('rawprofiles.png')
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


def synapse_periodisk_hydlim_skalert(Q,B,h0,l1,l2,tau1off,sigmaon,x1,nx,profilinterval,profilyn,deltax,dt,t_max,simnum):
    starttime=time.time()

    Q=float(Q)
    
    B=float(B)
    tau2off=tau1off
    tau1on=tau1off/3
    tau2on=tau2off/3
    taur= tau1on/tau1off
    
    dt0=dt
    x0=0
    lr=l1/l2
    
    nt=int(t_max/(dt0*profilinterval))
    
    tau1on=float(tau1on)
    tau2on=float(tau2on)
    tau1off=float(tau1off)
    tau2off=float(tau2off)
    sigmaon=float(sigmaon)
    
    l1=float(l1)
    l2=float(l2)
    h0=float(h0)
    
    #% Lag mesh
    dt=Constant(dt)
    deltax=(x1-x0)/nx

    mesh = RectangleMesh( Point(x0,x0), Point(x1, x1), nx, nx, diagonal='right')
    #mesh = generate_mesh(domain)
    #plot(mesh)
    class Periodic_sides(SubDomain):
        def inside(self, x, on_boundary):
            return bool((near(x[0], 0) or near(x[1], 0)) and (not ((near(x[0],0) and near(x[1],x1)) or (near(x[0],x1) and near(x[1],0)))) and on_boundary)

        def map(self, x, y):
            if near(x[0],x1) and near(x[1],x1):
                y[0] = x[0] - x1
                y[1] = x[1] - x1
            elif near(x[0],x1):
                y[0] = x[0] - x1
                y[1] = x[1]
            elif near(x[1],x1):
                y[0] = x[0]
                y[1] = x[1] - x1
            else:
                y[0]=-1000
                y[1]=-1000  
        
        
    #% Define mixed function space
    EL = FiniteElement('P',triangle,1)
    element = MixedElement([EL,EL,EL])
    VWS = FunctionSpace(mesh,element, constrained_domain=Periodic_sides())
    # Define initial condition 
    hpm_0 = Expression(('h0','0','0'), degree=2,x1=x1,h0=h0)

    # Define trial and test functions within function space
    v, w,s = TestFunctions(VWS)
    hpm = Function(VWS)
    h,p,m, = split(hpm)
    # Initialize solution in our function space at t=0
    hpm_n=project(hpm_0, VWS)
    h_n, p_n,m_n = split(hpm_n)
    # Make eta a function
    elmts = VectorElement('Lagrange', mesh.ufl_cell(), 1)
    etaV=FunctionSpace(mesh,elmts)
    eta=Function(etaV)
    
    ## Formulate the variational problem
    # Write out weak form of the equations with all terms on one side
    #
    #gammel med feilfaktorF=h_n*v*dx -h*v*dx-dt*Q*h**(3/2)*dot(grad(v),eta)*dx-dt*h**3*(1/(12))*dot(grad(p),grad(v))*dx +p*w*dx +B*dot(grad(w),grad(m))*dx -1*w*(exp(-((lr-h)/(sigmaon*lr))**2)/(exp(-((lr-h)/(sigmaon*lr))**2)+taur))*(h-lr)*dx - 1*w*lr*(exp(-((1-h)/(sigmaon*1))**2)/(exp(-((1-h)/(sigmaon*1))**2)+taur))*(h-1)*dx  +m*s*dx +dot(grad(s),grad(h))*dx   #+z1*c1_n*dx -z1*c1*dx +dt*z1*dot(grad(p),grad(c1))*(h*l1/mu)*dx +dt*(c10-c1)*z1*((1/tau1on)*exp(((l1-h)/(sigmaon*l1))**2))*dx -dt*c1*z1*((1/tau1off)*exp(((l1-h)/(sigmaoff*l1))**2))*dx -dt*D1*dot(grad(z1),grad(c1))*dx   +z2*c2_n*dx -z2*c2*dx +dt*z2*dot(grad(p),grad(c2))*(h*l2/mu)*dx +dt*(c20-c2)*z2*((1/tau2on)*exp(((l2-h)/(sigmaon*l2))**2))*dx -dt*c2*z2*((1/tau2off)*exp(((l2-h)/(sigmaoff*l2))**2))*dx -dt*D2*dot(grad(z2),grad(c2))*dx     
    F=h_n*v*dx -h*v*dx-dt*Q*h**(3/2)*dot(grad(v),eta)*dx-dt*h**3*dot(grad(p),grad(v))*dx +p*w*dx +B*dot(grad(w),grad(m))*dx -1*w*(exp(-((lr-h)/(sigmaon*lr))**2)/(exp(-((lr-h)/(sigmaon*lr))**2)+taur))*(h-lr)*dx - 1*w*lr*(exp(-((1-h)/(sigmaon*1))**2)/(exp(-((1-h)/(sigmaon*1))**2)+taur))*(h-1)*dx  +m*s*dx +dot(grad(s),grad(h))*dx   #+z1*c1_n*dx -z1*c1*dx +dt*z1*dot(grad(p),grad(c1))*(h*l1/mu)*dx +dt*(c10-c1)*z1*((1/tau1on)*exp(((l1-h)/(sigmaon*l1))**2))*dx -dt*c1*z1*((1/tau1off)*exp(((l1-h)/(sigmaoff*l1))**2))*dx -dt*D1*dot(grad(z1),grad(c1))*dx   +z2*c2_n*dx -z2*c2*dx +dt*z2*dot(grad(p),grad(c2))*(h*l2/mu)*dx +dt*(c20-c2)*z2*((1/tau2on)*exp(((l2-h)/(sigmaon*l2))**2))*dx -dt*c2*z2*((1/tau2off)*exp(((l2-h)/(sigmaoff*l2))**2))*dx -dt*D2*dot(grad(z2),grad(c2))*dx     

    J=derivative(F,hpm)

    problem=NonlinearVariationalProblem(F,hpm,J=J)
    solver=NonlinearVariationalSolver(problem)
    
    #% lag filer der data kan lagres
    t_series=open('t.txt','w')   # create and open text files in which to write down parameter values
    t_profiles=open('t_profil.txt','w')
    h_series=open('h.txt','w')
    xdmffile = XDMFFile('solution.xdmf') #create xdmf file for the solution
    xdmffile.parameters["flush_output"] = True
    profiles=np.zeros((nx+1,nx+1,nt+1))
    ts=np.zeros(nt)
    xs=np.zeros(nx+1)
    ys=np.zeros(nx+1)
    profts=np.zeros((nt+1))
    x=mesh.coordinates()
    xs=x[0:nx+1,0]
    ys=x[0:nx+1,0]
    nxd=nx
    xd=np.linspace(x0,x1,num=nxd)
    min=1000
    max=1000
    xmin=0
    for i1 in range(nxd-1):
        for i2 in range(nxd-1):
            val=hpm_n(xd[i1],xd[i2])[0]
            if val < min:
                min=val
                idmx=i1
                idmy=i2
    xmin=xd[idmx]
    ymin=xd[idmy]
    h_series.write(str(round(min,6)))       # write value
    h_series.write('\n') 


    def stochastic(Number,deltax,dt):
        mesh = UnitIntervalMesh(Number)        # lag mesh
        elms = FiniteElement('Lagrange', mesh.ufl_cell(), 1)  #
        U = FunctionSpace(mesh, elms)     # funksjonsrom,
        u = TrialFunction(U)
        v = TestFunction(U)
        eta = Function(U)
        stdevinv = deltax*np.sqrt(dt)
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
        
    #% registrer metadata
    simpars=open('simparametere.txt','w')
    simpars.write('nx = '+str(nx))
    simpars.write('\n')
    simpars.write('x0 = '+str(x0))
    simpars.write('\n')
    simpars.write('x1 = '+str(x1))
    simpars.write('\n')
    simpars.write('dt = '+str(dt0))
    simpars.write('\n')
    simpars.write('Q = '+str(Q))
    simpars.write('\n')
    simpars.write('B = '+str(B))
    simpars.write('\n')
    simpars.write('h_0 = '+str(h0))
    simpars.write('\n')

    #% Initialisér løsning
    F0=h*v*dx-h_n*v*dx +p*w*dx-p_n*w*dx   +m*s*dx -m_n*s*dx 
    solve(F0==0,hpm)
    
    #% Loop i tid
    t=0
    profts[0]=t
    _h0,_p0, _m0= hpm.split()
    xdmffile.write(_h0,t)
    t_series.write(str(round(t,6)))
    t_series.write('\n')
    t_profiles.write(str(round(t,6)))
    t_profiles.write('\n')

    for ix in range(nx+1):
        for iy in range(nx+1):
            profiles[ix,iy,0]=h0


    itno_old =0
    i=0
    iprof=0
    imax=-1000
    vals = np.zeros(2)
    klokk0=0
    klokk1=0
    klokk2=0
    while t<t_max:
        i+=1
        print('\n   \n')
        print( 'Q = '+str(Q)+', h0 = '+str(h0)+' -- Sim nummer '+str(simnum))
       
        klokk2=klokk1
        klokk1=klokk0
        klokk0=time.time()
        klokkdelt=np.mean([klokk0-klokk1,klokk1-klokk2])
        irem=t_max/dt0-i
        trem=klokkdelt*irem
        if i>=3:
            print('Estimated time remaining = '+str(datetime.timedelta(seconds=klokkdelt*irem)))
 
        # update current timestep
        t+=float(dt)                       # Time for which we will be computing solution
        print('Timestep number: '+str(i))#+'  /'+str(Nt))
        print('t = '+str(round(float(t),7)))#+'  /'+str(T))
        stochastic(len(eta.vector()),deltax,dt0)    # oppdater eta i tekstfil
        
        etaraw=np.loadtxt('u.txt') # last inn eta
        etaraw=etaraw[:,1]          # ta bare andre kolonne

        for ik in range(len(etaraw)-1):
            #print(ik)
            eta.vector()[ik] = etaraw[ik]         # sett vektorverdiene inn i funksjonen eta                         *********************
            if (np.sqrt(eta.vector()[ik]**2-etaraw[ik]**2)) != 0:
                print('Fail to assign new values to eta')


        # solve the variational problem to get new u at current timestep
        itno,conv =solver.solve()
        
        _h,_p, _m= hpm.split()
    
        hpm_n.assign(hpm)
        t_series.write(str(t))
        t_series.write('\n')
        

        if profilyn==1 and i/profilinterval==int(i/profilinterval):
            iprof+=1
            t_profiles.write(str(round(t,6)))
            t_profiles.write('\n')
            profts[iprof]=t
            for ix in range(nx+1):
                for iy in range(nx+1):
                    profiles[ix,iy,iprof]=_h(xs[ix],ys[iy])
            xdmffile.write(_h,t)
            
        min=1000
        for i1 in range(nxd-1):
            for i2 in range(nxd-1):
                val=hpm_n(xd[i1],xd[i2])[0]
                if val < min:
                    min=val
                    idmx=i1
                    idmy=i2
        h_series.write(str(round(min,6)))       # write value
        h_series.write('\n') 
    #%
    h_series.close()
    t_series.close()
    t_profiles.close()

    endtime=time.time()
    print('Elapsed time = '+str(datetime.timedelta(seconds=endtime-starttime)))
    simpars.write('Elapsed real time = '+str(datetime.timedelta(seconds=endtime-starttime)))
    simpars.write('\n')
    simpars.write('# of iterations = '+str(i))
    simpars.write('\n')
    simpars.write('Simulation end time = '+str(t))
    simpars.close()

    #%
    np.savez('profiles.npz',profiles=profiles,xs=xs,ys=ys,profts=profts)


def synapse_periodisk_hydlim_skalert_deterministisk(Q,B,h0,l1,l2,tau1off,sigmaon,x1,nx,profilinterval,profilyn,deltax,dt,t_max,simnum):
    starttime=time.time()

    Q=float(Q)
    
    B=float(B)
    tau2off=tau1off
    tau1on=tau1off/3
    tau2on=tau2off/3
    taur= tau1on/tau1off
    
    dt0=dt
    x0=0
    lr=l1/l2
    
    nt=int(t_max/(dt0*profilinterval))
    
    tau1on=float(tau1on)
    tau2on=float(tau2on)
    tau1off=float(tau1off)
    tau2off=float(tau2off)
    sigmaon=float(sigmaon)
    
    l1=float(l1)
    l2=float(l2)
    h0=float(h0)
    
    #% Lag mesh
    dt=Constant(dt)
    deltax=(x1-x0)/nx

    mesh = RectangleMesh( Point(x0,x0), Point(x1, x1), nx, nx, diagonal='right')
    #mesh = generate_mesh(domain)
    #plot(mesh)
    class Periodic_sides(SubDomain):
        def inside(self, x, on_boundary):
            return bool((near(x[0], 0) or near(x[1], 0)) and (not ((near(x[0],0) and near(x[1],x1)) or (near(x[0],x1) and near(x[1],0)))) and on_boundary)

        def map(self, x, y):
            if near(x[0],x1) and near(x[1],x1):
                y[0] = x[0] - x1
                y[1] = x[1] - x1
            elif near(x[0],x1):
                y[0] = x[0] - x1
                y[1] = x[1]
            elif near(x[1],x1):
                y[0] = x[0]
                y[1] = x[1] - x1
            else:
                y[0]=-1000
                y[1]=-1000  
        
        
    #% Define mixed function space
    EL = FiniteElement('P',triangle,1)
    element = MixedElement([EL,EL,EL])
    VWS = FunctionSpace(mesh,element, constrained_domain=Periodic_sides())
    # Define initial condition 
    hpm_0 = Expression(('(pow(x[0]-x1/3,2)+pow(x[1]-x1/3,2))<pow(10*x1/36,2) ? lr :(pow(x[0]-4*x1/5,2)+pow(x[1]-4*x1/5,2))<pow(3*x1/18,2) ? lr:1','0','0'), degree=2,x1=x1,h0=h0,lr=lr)

    # Define trial and test functions within function space
    v, w,s = TestFunctions(VWS)
    hpm = Function(VWS)
    h,p,m, = split(hpm)
    # Initialize solution in our function space at t=0
    hpm_n=project(hpm_0, VWS)
    h_n, p_n,m_n = split(hpm_n)
    # Make eta a function
    elmts = VectorElement('Lagrange', mesh.ufl_cell(), 1)
    etaV=FunctionSpace(mesh,elmts)
    eta=Function(etaV)
    
    ## Formulate the variational problem
    # Write out weak form of the equations with all terms on one side
    #
    #gammel med feilfaktorF=h_n*v*dx -h*v*dx-dt*Q*h**(3/2)*dot(grad(v),eta)*dx-dt*h**3*(1/(12))*dot(grad(p),grad(v))*dx +p*w*dx +B*dot(grad(w),grad(m))*dx -1*w*(exp(-((lr-h)/(sigmaon*lr))**2)/(exp(-((lr-h)/(sigmaon*lr))**2)+taur))*(h-lr)*dx - 1*w*lr*(exp(-((1-h)/(sigmaon*1))**2)/(exp(-((1-h)/(sigmaon*1))**2)+taur))*(h-1)*dx  +m*s*dx +dot(grad(s),grad(h))*dx   #+z1*c1_n*dx -z1*c1*dx +dt*z1*dot(grad(p),grad(c1))*(h*l1/mu)*dx +dt*(c10-c1)*z1*((1/tau1on)*exp(((l1-h)/(sigmaon*l1))**2))*dx -dt*c1*z1*((1/tau1off)*exp(((l1-h)/(sigmaoff*l1))**2))*dx -dt*D1*dot(grad(z1),grad(c1))*dx   +z2*c2_n*dx -z2*c2*dx +dt*z2*dot(grad(p),grad(c2))*(h*l2/mu)*dx +dt*(c20-c2)*z2*((1/tau2on)*exp(((l2-h)/(sigmaon*l2))**2))*dx -dt*c2*z2*((1/tau2off)*exp(((l2-h)/(sigmaoff*l2))**2))*dx -dt*D2*dot(grad(z2),grad(c2))*dx     
    F=h_n*v*dx -h*v*dx-dt*h**3*dot(grad(p),grad(v))*dx +p*w*dx +B*dot(grad(w),grad(m))*dx -1*w*(exp(-((lr-h)/(sigmaon*lr))**2)/(exp(-((lr-h)/(sigmaon*lr))**2)+taur))*(h-lr)*dx - 1*w*lr*(exp(-((1-h)/(sigmaon*1))**2)/(exp(-((1-h)/(sigmaon*1))**2)+taur))*(h-1)*dx  +m*s*dx +dot(grad(s),grad(h))*dx   #+z1*c1_n*dx -z1*c1*dx +dt*z1*dot(grad(p),grad(c1))*(h*l1/mu)*dx +dt*(c10-c1)*z1*((1/tau1on)*exp(((l1-h)/(sigmaon*l1))**2))*dx -dt*c1*z1*((1/tau1off)*exp(((l1-h)/(sigmaoff*l1))**2))*dx -dt*D1*dot(grad(z1),grad(c1))*dx   +z2*c2_n*dx -z2*c2*dx +dt*z2*dot(grad(p),grad(c2))*(h*l2/mu)*dx +dt*(c20-c2)*z2*((1/tau2on)*exp(((l2-h)/(sigmaon*l2))**2))*dx -dt*c2*z2*((1/tau2off)*exp(((l2-h)/(sigmaoff*l2))**2))*dx -dt*D2*dot(grad(z2),grad(c2))*dx     

    J=derivative(F,hpm)

    problem=NonlinearVariationalProblem(F,hpm,J=J)
    solver=NonlinearVariationalSolver(problem)
    
    #% lag filer der data kan lagres
    t_series=open('t.txt','w')   # create and open text files in which to write down parameter values
    t_profiles=open('t_profil.txt','w')
    h_series=open('h.txt','w')
    xdmffile = XDMFFile('solution.xdmf') #create xdmf file for the solution
    xdmffile.parameters["flush_output"] = True
    profiles=np.zeros((nx+1,nx+1,nt+1))
    ts=np.zeros(nt)
    xs=np.zeros(nx+1)
    ys=np.zeros(nx+1)
    profts=np.zeros((nt+1))
    x=mesh.coordinates()
    xs=x[0:nx+1,0]
    ys=x[0:nx+1,0]
    nxd=nx
    xd=np.linspace(x0,x1,num=nxd)
    min=1000
    max=1000
    xmin=0
    for i1 in range(nxd-1):
        for i2 in range(nxd-1):
            val=hpm_n(xd[i1],xd[i2])[0]
            if val < min:
                min=val
                idmx=i1
                idmy=i2
    xmin=xd[idmx]
    ymin=xd[idmy]
    h_series.write(str(round(min,6)))       # write value
    h_series.write('\n') 


# =============================================================================
#     def stochastic(Number,deltax,dt):
#         mesh = UnitIntervalMesh(Number)        # lag mesh
#         elms = FiniteElement('Lagrange', mesh.ufl_cell(), 1)  #
#         U = FunctionSpace(mesh, elms)     # funksjonsrom,
#         u = TrialFunction(U)
#         v = TestFunction(U)
#         eta = Function(U)
#         stdevinv = deltax*np.sqrt(dt)
#         stdev=1/stdevinv
#         def update_random(random=eta):    # funksjon
#             random.vector().set_local(np.random.normal(0, stdev, random.vector().local_size()))
#             random.vector().apply('insert')
#         eq = inner(u, v)*dx - inner(eta, v)*dx
#         u = Function(U)
#         update_random()
#         solve(lhs(eq) == rhs(eq), u)
#         outfile = open('u.txt', 'w')
#         for i in range(len(mesh.coordinates())):
#             outfile.write('%f\t %f\n' %(mesh.coordinates()[i], u(mesh.coordinates()[i])))
#         outfile.close()
#         
# =============================================================================
    #% registrer metadata
    simpars=open('simparametere.txt','w')
    simpars.write('nx = '+str(nx))
    simpars.write('\n')
    simpars.write('x0 = '+str(x0))
    simpars.write('\n')
    simpars.write('x1 = '+str(x1))
    simpars.write('\n')
    simpars.write('dt = '+str(dt0))
    simpars.write('\n')
    simpars.write('Q = '+str(Q))
    simpars.write('\n')
    simpars.write('B = '+str(B))
    simpars.write('\n')
    simpars.write('h_0 = '+str(h0))
    simpars.write('\n')

    #% Initialisér løsning
    F0=h*v*dx-h_n*v*dx +p*w*dx-p_n*w*dx   +m*s*dx -m_n*s*dx 
    solve(F0==0,hpm)
    
    #% Loop i tid
    t=0
    profts[0]=t
    _h0,_p0, _m0= hpm.split()
    xdmffile.write(_h0,t)
    t_series.write(str(round(t,6)))
    t_series.write('\n')
    t_profiles.write(str(round(t,6)))
    t_profiles.write('\n')

    for ix in range(nx+1):
        for iy in range(nx+1):
            profiles[ix,iy,0]=h0


    itno_old =0
    i=0
    iprof=0
    imax=-1000
    vals = np.zeros(2)
    klokk0=0
    klokk1=0
    klokk2=0
    while t<t_max:
        i+=1
        print('\n   \n')
        print( 'Q = '+str(Q)+', h0 = '+str(h0)+' -- Sim nummer '+str(simnum))
       
        klokk2=klokk1
        klokk1=klokk0
        klokk0=time.time()
        klokkdelt=np.mean([klokk0-klokk1,klokk1-klokk2])
        irem=t_max/dt0-i
        trem=klokkdelt*irem
        if i>=3:
            print('Estimated time remaining = '+str(datetime.timedelta(seconds=klokkdelt*irem)))
        
        # update current timestep
        t+=float(dt)                       # Time for which we will be computing solution
        print('Timestep number: '+str(i))#+'  /'+str(Nt))
        print('t = '+str(round(float(t),7)))#+'  /'+str(T))
# =============================================================================
#         stochastic(len(eta.vector()),deltax,dt0)    # oppdater eta i tekstfil
#         
# =============================================================================
        #etaraw=np.loadtxt('u.txt') # last inn eta
        #etaraw=etaraw[:,1]          # ta bare andre kolonne

# =============================================================================
#         for ik in range(len(etaraw)-1):
#             #print(ik)
#             eta.vector()[ik] = etaraw[ik]         # sett vektorverdiene inn i funksjonen eta                         *********************
#             if (np.sqrt(eta.vector()[ik]**2-etaraw[ik]**2)) != 0:
#                 print('Fail to assign new values to eta')
# 
# =============================================================================

        # solve the variational problem to get new u at current timestep
        itno,conv =solver.solve()
        
        _h,_p, _m= hpm.split()
    
        hpm_n.assign(hpm)
        t_series.write(str(t))
        t_series.write('\n')
        

        if profilyn==1 and i/profilinterval==int(i/profilinterval):
            iprof+=1
            t_profiles.write(str(round(t,6)))
            t_profiles.write('\n')
            profts[iprof]=t
            for ix in range(nx+1):
                for iy in range(nx+1):
                    profiles[ix,iy,iprof]=_h(xs[ix],ys[iy])
            xdmffile.write(_h,t)
            
        min=1000
        for i1 in range(nxd-1):
            for i2 in range(nxd-1):
                val=hpm_n(xd[i1],xd[i2])[0]
                if val < min:
                    min=val
                    idmx=i1
                    idmy=i2
        h_series.write(str(round(min,6)))       # write value
        h_series.write('\n') 
    #%
    h_series.close()
    t_series.close()
    t_profiles.close()

    endtime=time.time()
    print('Elapsed time = '+str(datetime.timedelta(seconds=endtime-starttime)))
    simpars.write('Elapsed real time = '+str(datetime.timedelta(seconds=endtime-starttime)))
    simpars.write('\n')
    simpars.write('# of iterations = '+str(i))
    simpars.write('\n')
    simpars.write('Simulation end time = '+str(t))
    simpars.close()

    #%
    np.savez('profiles.npz',profiles=profiles,xs=xs,ys=ys,profts=profts)

def synapse_periodisk_hydlim_skalert_deterministisk_mange(Q,B,h0,l1,l2,tau1off,sigmaon,x1,nx,profilinterval,profilyn,deltax,dt,t_max,simnum):
    starttime=time.time()

    Q=float(Q)
    
    B=float(B)
    tau2off=tau1off
    tau1on=tau1off/3
    tau2on=tau2off/3
    taur= tau1on/tau1off
    
    dt0=dt
    x0=0
    lr=l1/l2
    
    nt=int(t_max/(dt0*profilinterval))
    
    tau1on=float(tau1on)
    tau2on=float(tau2on)
    tau1off=float(tau1off)
    tau2off=float(tau2off)
    sigmaon=float(sigmaon)
    
    l1=float(l1)
    l2=float(l2)
    h0=float(h0)
    
    #% Lag mesh
    dt=Constant(dt)
    deltax=(x1-x0)/nx

    mesh = RectangleMesh( Point(x0,x0), Point(x1, x1), nx, nx, diagonal='right')
    #mesh = generate_mesh(domain)
    #plot(mesh)
    class Periodic_sides(SubDomain):
        def inside(self, x, on_boundary):
            return bool((near(x[0], 0) or near(x[1], 0)) and (not ((near(x[0],0) and near(x[1],x1)) or (near(x[0],x1) and near(x[1],0)))) and on_boundary)

        def map(self, x, y):
            if near(x[0],x1) and near(x[1],x1):
                y[0] = x[0] - x1
                y[1] = x[1] - x1
            elif near(x[0],x1):
                y[0] = x[0] - x1
                y[1] = x[1]
            elif near(x[1],x1):
                y[0] = x[0]
                y[1] = x[1] - x1
            else:
                y[0]=-1000
                y[1]=-1000  
    
    nrand=30
    randomx =0.04+0.92*np.random.rand(nrand)
    randomy =0.04+0.92*np.random.rand(nrand)
    randomsize=np.random.normal(0.042,0.0035 , nrand)
        
    #% Define mixed function space
    EL = FiniteElement('P',triangle,1)
    element = MixedElement([EL,EL,EL])
    VWS = FunctionSpace(mesh,element, constrained_domain=Periodic_sides())
    # Define initial condition 
    hpm_0 = Expression(('(pow(x[0]-x1*'+str(randomx[0])+',2)+pow(x[1]-x1*'+str(randomy[0])+',2))<pow('+str(randomsize[0])+'*x1,2) ? lr :(pow(x[0]-x1*'+str(randomx[1])+',2)+pow(x[1]-x1*'+str(randomy[1])+',2))<pow('+str(randomsize[1])+'*x1,2) ? lr :(pow(x[0]-x1*'+str(randomx[2])+',2)+pow(x[1]-x1*'+str(randomy[2])+',2))<pow('+str(randomsize[2])+'*x1,2) ? lr :(pow(x[0]-x1*'+str(randomx[3])+',2)+pow(x[1]-x1*'+str(randomy[3])+',2))<pow('+str(randomsize[3])+'*x1,2) ? lr :(pow(x[0]-x1*'+str(randomx[4])+',2)+pow(x[1]-x1*'+str(randomy[4])+',2))<pow('+str(randomsize[4])+'*x1,2) ? lr :(pow(x[0]-x1*'+str(randomx[5])+',2)+pow(x[1]-x1*'+str(randomy[5])+',2))<pow('+str(randomsize[5])+'*x1,2) ? lr :(pow(x[0]-x1*'+str(randomx[6])+',2)+pow(x[1]-x1*'+str(randomy[6])+',2))<pow('+str(randomsize[6])+'*x1,2) ? lr :(pow(x[0]-x1*'+str(randomx[7])+',2)+pow(x[1]-x1*'+str(randomy[7])+',2))<pow('+str(randomsize[7])+'*x1,2) ? lr :(pow(x[0]-x1*'+str(randomx[8])+',2)+pow(x[1]-x1*'+str(randomy[8])+',2))<pow('+str(randomsize[8])+'*x1,2) ? lr :(pow(x[0]-x1*'+str(randomx[9])+',2)+pow(x[1]-x1*'+str(randomy[9])+',2))<pow('+str(randomsize[9])+'*x1,2) ? lr :(pow(x[0]-x1*'+str(randomx[10])+',2)+pow(x[1]-x1*'+str(randomy[10])+',2))<pow('+str(randomsize[10])+'*x1,2) ? lr :(pow(x[0]-x1*'+str(randomx[11])+',2)+pow(x[1]-x1*'+str(randomy[11])+',2))<pow('+str(randomsize[11])+'*x1,2) ? lr :(pow(x[0]-x1*'+str(randomx[12])+',2)+pow(x[1]-x1*'+str(randomy[12])+',2))<pow('+str(randomsize[12])+'*x1,2) ? lr :(pow(x[0]-x1*'+str(randomx[13])+',2)+pow(x[1]-x1*'+str(randomy[13])+',2))<pow('+str(randomsize[13])+'*x1,2) ? lr :(pow(x[0]-x1*'+str(randomx[14])+',2)+pow(x[1]-x1*'+str(randomy[14])+',2))<pow('+str(randomsize[14])+'*x1,2) ? lr :(pow(x[0]-x1*'+str(randomx[15])+',2)+pow(x[1]-x1*'+str(randomy[15])+',2))<pow('+str(randomsize[15])+'*x1,2) ? lr :(pow(x[0]-x1*'+str(randomx[16])+',2)+pow(x[1]-x1*'+str(randomy[16])+',2))<pow('+str(randomsize[16])+'*x1,2) ? lr :(pow(x[0]-x1*'+str(randomx[17])+',2)+pow(x[1]-x1*'+str(randomy[17])+',2))<pow('+str(randomsize[17])+'*x1,2) ? lr :(pow(x[0]-x1*'+str(randomx[18])+',2)+pow(x[1]-x1*'+str(randomy[18])+',2))<pow('+str(randomsize[18])+'*x1,2) ? lr :(pow(x[0]-x1*'+str(randomx[19])+',2)+pow(x[1]-x1*'+str(randomy[19])+',2))<pow('+str(randomsize[19])+'*x1,2) ? lr :(pow(x[0]-x1*'+str(randomx[20])+',2)+pow(x[1]-x1*'+str(randomy[20])+',2))<pow('+str(randomsize[20])+'*x1,2) ? lr :(pow(x[0]-x1*'+str(randomx[21])+',2)+pow(x[1]-x1*'+str(randomy[21])+',2))<pow('+str(randomsize[21])+'*x1,2) ? lr :(pow(x[0]-x1*'+str(randomx[22])+',2)+pow(x[1]-x1*'+str(randomy[22])+',2))<pow('+str(randomsize[22])+'*x1,2) ? lr :(pow(x[0]-x1*'+str(randomx[23])+',2)+pow(x[1]-x1*'+str(randomy[23])+',2))<pow('+str(randomsize[23])+'*x1,2) ? lr :(pow(x[0]-x1*'+str(randomx[24])+',2)+pow(x[1]-x1*'+str(randomy[24])+',2))<pow('+str(randomsize[24])+'*x1,2) ? lr :(pow(x[0]-x1*'+str(randomx[25])+',2)+pow(x[1]-x1*'+str(randomy[25])+',2))<pow('+str(randomsize[25])+'*x1,2) ? lr :(pow(x[0]-x1*'+str(randomx[26])+',2)+pow(x[1]-x1*'+str(randomy[26])+',2))<pow('+str(randomsize[26])+'*x1,2) ? lr :1','0','0'), degree=2,x1=x1,h0=h0,lr=lr)

    # Define trial and test functions within function space
    v, w,s = TestFunctions(VWS)
    hpm = Function(VWS)
    h,p,m, = split(hpm)
    # Initialize solution in our function space at t=0
    hpm_n=project(hpm_0, VWS)
    h_n, p_n,m_n = split(hpm_n)
    # Make eta a function
    elmts = VectorElement('Lagrange', mesh.ufl_cell(), 1)
    etaV=FunctionSpace(mesh,elmts)
    eta=Function(etaV)
    
    ## Formulate the variational problem
    # Write out weak form of the equations with all terms on one side
    #
    #gammel med feilfaktorF=h_n*v*dx -h*v*dx-dt*Q*h**(3/2)*dot(grad(v),eta)*dx-dt*h**3*(1/(12))*dot(grad(p),grad(v))*dx +p*w*dx +B*dot(grad(w),grad(m))*dx -1*w*(exp(-((lr-h)/(sigmaon*lr))**2)/(exp(-((lr-h)/(sigmaon*lr))**2)+taur))*(h-lr)*dx - 1*w*lr*(exp(-((1-h)/(sigmaon*1))**2)/(exp(-((1-h)/(sigmaon*1))**2)+taur))*(h-1)*dx  +m*s*dx +dot(grad(s),grad(h))*dx   #+z1*c1_n*dx -z1*c1*dx +dt*z1*dot(grad(p),grad(c1))*(h*l1/mu)*dx +dt*(c10-c1)*z1*((1/tau1on)*exp(((l1-h)/(sigmaon*l1))**2))*dx -dt*c1*z1*((1/tau1off)*exp(((l1-h)/(sigmaoff*l1))**2))*dx -dt*D1*dot(grad(z1),grad(c1))*dx   +z2*c2_n*dx -z2*c2*dx +dt*z2*dot(grad(p),grad(c2))*(h*l2/mu)*dx +dt*(c20-c2)*z2*((1/tau2on)*exp(((l2-h)/(sigmaon*l2))**2))*dx -dt*c2*z2*((1/tau2off)*exp(((l2-h)/(sigmaoff*l2))**2))*dx -dt*D2*dot(grad(z2),grad(c2))*dx     
    F=h_n*v*dx -h*v*dx-dt*h**3*dot(grad(p),grad(v))*dx +p*w*dx +B*dot(grad(w),grad(m))*dx -1*w*(exp(-((lr-h)/(sigmaon*lr))**2)/(exp(-((lr-h)/(sigmaon*lr))**2)+taur))*(h-lr)*dx - 1*w*lr*(exp(-((1-h)/(sigmaon*1))**2)/(exp(-((1-h)/(sigmaon*1))**2)+taur))*(h-1)*dx  +m*s*dx +dot(grad(s),grad(h))*dx   #+z1*c1_n*dx -z1*c1*dx +dt*z1*dot(grad(p),grad(c1))*(h*l1/mu)*dx +dt*(c10-c1)*z1*((1/tau1on)*exp(((l1-h)/(sigmaon*l1))**2))*dx -dt*c1*z1*((1/tau1off)*exp(((l1-h)/(sigmaoff*l1))**2))*dx -dt*D1*dot(grad(z1),grad(c1))*dx   +z2*c2_n*dx -z2*c2*dx +dt*z2*dot(grad(p),grad(c2))*(h*l2/mu)*dx +dt*(c20-c2)*z2*((1/tau2on)*exp(((l2-h)/(sigmaon*l2))**2))*dx -dt*c2*z2*((1/tau2off)*exp(((l2-h)/(sigmaoff*l2))**2))*dx -dt*D2*dot(grad(z2),grad(c2))*dx     

    J=derivative(F,hpm)

    problem=NonlinearVariationalProblem(F,hpm,J=J)
    solver=NonlinearVariationalSolver(problem)
    
    #% lag filer der data kan lagres
    t_series=open('t.txt','w')   # create and open text files in which to write down parameter values
    t_profiles=open('t_profil.txt','w')
    h_series=open('h.txt','w')
    xdmffile = XDMFFile('solution.xdmf') #create xdmf file for the solution
    xdmffile.parameters["flush_output"] = True
    profiles=np.zeros((nx+1,nx+1,nt+1))
    ts=np.zeros(nt)
    xs=np.zeros(nx+1)
    ys=np.zeros(nx+1)
    profts=np.zeros((nt+1))
    x=mesh.coordinates()
    xs=x[0:nx+1,0]
    ys=x[0:nx+1,0]
    nxd=nx
    xd=np.linspace(x0,x1,num=nxd)
    min=1000
    max=1000
    xmin=0
    for i1 in range(nxd-1):
        for i2 in range(nxd-1):
            val=hpm_n(xd[i1],xd[i2])[0]
            if val < min:
                min=val
                idmx=i1
                idmy=i2
    xmin=xd[idmx]
    ymin=xd[idmy]
    h_series.write(str(round(min,6)))       # write value
    h_series.write('\n') 


# =============================================================================
#     def stochastic(Number,deltax,dt):
#         mesh = UnitIntervalMesh(Number)        # lag mesh
#         elms = FiniteElement('Lagrange', mesh.ufl_cell(), 1)  #
#         U = FunctionSpace(mesh, elms)     # funksjonsrom,
#         u = TrialFunction(U)
#         v = TestFunction(U)
#         eta = Function(U)
#         stdevinv = deltax*np.sqrt(dt)
#         stdev=1/stdevinv
#         def update_random(random=eta):    # funksjon
#             random.vector().set_local(np.random.normal(0, stdev, random.vector().local_size()))
#             random.vector().apply('insert')
#         eq = inner(u, v)*dx - inner(eta, v)*dx
#         u = Function(U)
#         update_random()
#         solve(lhs(eq) == rhs(eq), u)
#         outfile = open('u.txt', 'w')
#         for i in range(len(mesh.coordinates())):
#             outfile.write('%f\t %f\n' %(mesh.coordinates()[i], u(mesh.coordinates()[i])))
#         outfile.close()
#         
# =============================================================================
    #% registrer metadata
    simpars=open('simparametere.txt','w')
    simpars.write('nx = '+str(nx))
    simpars.write('\n')
    simpars.write('x0 = '+str(x0))
    simpars.write('\n')
    simpars.write('x1 = '+str(x1))
    simpars.write('\n')
    simpars.write('dt = '+str(dt0))
    simpars.write('\n')
    simpars.write('Q = '+str(Q))
    simpars.write('\n')
    simpars.write('B = '+str(B))
    simpars.write('\n')
    simpars.write('h_0 = '+str(h0))
    simpars.write('\n')

    #% Initialisér løsning
    F0=h*v*dx-h_n*v*dx +p*w*dx-p_n*w*dx   +m*s*dx -m_n*s*dx 
    solve(F0==0,hpm)
    
    #% Loop i tid
    t=0
    profts[0]=t
    _h0,_p0, _m0= hpm.split()
    xdmffile.write(_h0,t)
    t_series.write(str(round(t,6)))
    t_series.write('\n')
    t_profiles.write(str(round(t,6)))
    t_profiles.write('\n')

    for ix in range(nx+1):
        for iy in range(nx+1):
            profiles[ix,iy,0]=h0


    itno_old =0
    i=0
    iprof=0
    imax=-1000
    vals = np.zeros(2)
    klokk0=0
    klokk1=0
    klokk2=0
    while t<t_max:
        i+=1
        print('\n   \n')
        print( 'Q = '+str(Q)+', h0 = '+str(h0)+' -- Sim nummer '+str(simnum))
       
        klokk2=klokk1
        klokk1=klokk0
        klokk0=time.time()
        klokkdelt=np.mean([klokk0-klokk1,klokk1-klokk2])
        irem=t_max/dt0-i
        trem=klokkdelt*irem
        print('Estimated time remaining = '+str(datetime.timedelta(seconds=klokkdelt*irem)))
        
        # update current timestep
        t+=float(dt)                       # Time for which we will be computing solution
        print('Timestep number: '+str(i))#+'  /'+str(Nt))
        print('t = '+str(round(float(t),7)))#+'  /'+str(T))
# =============================================================================
#         stochastic(len(eta.vector()),deltax,dt0)    # oppdater eta i tekstfil
#         
# =============================================================================
        #etaraw=np.loadtxt('u.txt') # last inn eta
        #etaraw=etaraw[:,1]          # ta bare andre kolonne

# =============================================================================
#         for ik in range(len(etaraw)-1):
#             #print(ik)
#             eta.vector()[ik] = etaraw[ik]         # sett vektorverdiene inn i funksjonen eta                         *********************
#             if (np.sqrt(eta.vector()[ik]**2-etaraw[ik]**2)) != 0:
#                 print('Fail to assign new values to eta')
# 
# =============================================================================

        # solve the variational problem to get new u at current timestep
        itno,conv =solver.solve()
        
        _h,_p, _m= hpm.split()
    
        hpm_n.assign(hpm)
        t_series.write(str(t))
        t_series.write('\n')
        

        if profilyn==1 and i/profilinterval==int(i/profilinterval):
            iprof+=1
            t_profiles.write(str(round(t,6)))
            t_profiles.write('\n')
            profts[iprof]=t
            for ix in range(nx+1):
                for iy in range(nx+1):
                    profiles[ix,iy,iprof]=_h(xs[ix],ys[iy])
            xdmffile.write(_h,t)
            
        min=1000
        for i1 in range(nxd-1):
            for i2 in range(nxd-1):
                val=hpm_n(xd[i1],xd[i2])[0]
                if val < min:
                    min=val
                    idmx=i1
                    idmy=i2
        h_series.write(str(round(min,6)))       # write value
        h_series.write('\n') 
    #%
    h_series.close()
    t_series.close()
    t_profiles.close()

    endtime=time.time()
    print('Elapsed time = '+str(datetime.timedelta(seconds=endtime-starttime)))
    simpars.write('Elapsed real time = '+str(datetime.timedelta(seconds=endtime-starttime)))
    simpars.write('\n')
    simpars.write('# of iterations = '+str(i))
    simpars.write('\n')
    simpars.write('Simulation end time = '+str(t))
    simpars.close()

    #%
    np.savez('profiles.npz',profiles=profiles,xs=xs,ys=ys,profts=profts)
    
def synapse_periodisk_hydlim_skalert_deterministisk_grid(Q,B,h0,l1,l2,tau1off,sigmaon,x1,nx,profilinterval,profilyn,deltax,dt,t_max,simnum):
    starttime=time.time()

    Q=float(Q)
    
    B=float(B)
    tau2off=tau1off
    tau1on=tau1off/3
    tau2on=tau2off/3
    taur= tau1on/tau1off
    
    dt0=dt
    x0=0
    lr=l1/l2
    
    nt=int(t_max/(dt0*profilinterval))
    
    tau1on=float(tau1on)
    tau2on=float(tau2on)
    tau1off=float(tau1off)
    tau2off=float(tau2off)
    sigmaon=float(sigmaon)
    
    l1=float(l1)
    l2=float(l2)
    h0=float(h0)
    
    #% Lag mesh
    dt=Constant(dt)
    deltax=(x1-x0)/nx

    mesh = RectangleMesh( Point(x0,x0), Point(x1, x1), nx, nx, diagonal='right')
    #mesh = generate_mesh(domain)
    #plot(mesh)
    class Periodic_sides(SubDomain):
        def inside(self, x, on_boundary):
            return bool((near(x[0], 0) or near(x[1], 0)) and (not ((near(x[0],0) and near(x[1],x1)) or (near(x[0],x1) and near(x[1],0)))) and on_boundary)

        def map(self, x, y):
            if near(x[0],x1) and near(x[1],x1):
                y[0] = x[0] - x1
                y[1] = x[1] - x1
            elif near(x[0],x1):
                y[0] = x[0] - x1
                y[1] = x[1]
            elif near(x[1],x1):
                y[0] = x[0]
                y[1] = x[1] - x1
            else:
                y[0]=-1000
                y[1]=-1000  
    
    nrand=25
    #randomx =0.04+0.92*np.random.rand(nrand)
    #randomy =0.04+0.92*np.random.rand(nrand)
    randomsize=np.random.normal(0.06,0.002 , nrand)
        
    #% Define mixed function space
    EL = FiniteElement('P',triangle,1)
    element = MixedElement([EL,EL,EL])
    VWS = FunctionSpace(mesh,element, constrained_domain=Periodic_sides())
    # Define initial condition 
    h0=1.04
    hpm_0 = Expression(('(pow(x[0]-x1*0.1,2)+pow(x[1]-x1*0.1,2))<pow('+str(randomsize[0])+'*x1,2) ? lr :(pow(x[0]-x1*0.1,2)+pow(x[1]-x1*0.3,2))<pow('+str(randomsize[1])+'*x1,2) ? lr :(pow(x[0]-x1*0.1,2)+pow(x[1]-x1*0.5,2))<pow('+str(randomsize[2])+'*x1,2) ? lr :(pow(x[0]-x1*0.1,2)+pow(x[1]-x1*0.7,2))<pow('+str(randomsize[3])+'*x1,2) ? lr :(pow(x[0]-x1*0.1,2)+pow(x[1]-x1*0.9,2))<pow('+str(randomsize[4])+'*x1,2) ? lr :(pow(x[0]-x1*0.3,2)+pow(x[1]-x1*0.1,2))<pow('+str(randomsize[5])+'*x1,2) ? lr :(pow(x[0]-x1*0.3,2)+pow(x[1]-x1*0.3,2))<pow('+str(randomsize[6])+'*x1,2) ? lr :(pow(x[0]-x1*0.3,2)+pow(x[1]-x1*0.5,2))<pow('+str(randomsize[7])+'*x1,2) ? lr :(pow(x[0]-x1*0.3,2)+pow(x[1]-x1*0.7,2))<pow('+str(randomsize[8])+'*x1,2) ? lr :(pow(x[0]-x1*0.3,2)+pow(x[1]-x1*0.9,2))<pow('+str(randomsize[9])+'*x1,2) ? lr :(pow(x[0]-x1*0.5,2)+pow(x[1]-x1*0.1,2))<pow('+str(randomsize[10])+'*x1,2) ? lr :(pow(x[0]-x1*0.5,2)+pow(x[1]-x1*0.3,2))<pow('+str(randomsize[11])+'*x1,2) ? lr :(pow(x[0]-x1*0.5,2)+pow(x[1]-x1*0.5,2))<pow('+str(randomsize[12])+'*x1,2) ? lr :(pow(x[0]-x1*0.5,2)+pow(x[1]-x1*0.7,2))<pow('+str(randomsize[13])+'*x1,2) ? lr :(pow(x[0]-x1*0.5,2)+pow(x[1]-x1*0.9,2))<pow('+str(randomsize[14])+'*x1,2) ? lr :(pow(x[0]-x1*0.7,2)+pow(x[1]-x1*0.1,2))<pow('+str(randomsize[15])+'*x1,2) ? lr :(pow(x[0]-x1*0.7,2)+pow(x[1]-x1*0.3,2))<pow('+str(randomsize[16])+'*x1,2) ? lr :(pow(x[0]-x1*0.7,2)+pow(x[1]-x1*0.5,2))<pow('+str(randomsize[17])+'*x1,2) ? lr :(pow(x[0]-x1*0.7,2)+pow(x[1]-x1*0.7,2))<pow('+str(randomsize[18])+'*x1,2) ? lr :(pow(x[0]-x1*0.7,2)+pow(x[1]-x1*0.9,2))<pow('+str(randomsize[19])+'*x1,2) ? lr :(pow(x[0]-x1*0.9,2)+pow(x[1]-x1*0.1,2))<pow('+str(randomsize[20])+'*x1,2) ? lr :(pow(x[0]-x1*0.9,2)+pow(x[1]-x1*0.3,2))<pow('+str(randomsize[21])+'*x1,2) ? lr :(pow(x[0]-x1*0.9,2)+pow(x[1]-x1*0.5,2))<pow('+str(randomsize[22])+'*x1,2) ? lr :(pow(x[0]-x1*0.9,2)+pow(x[1]-x1*0.7,2))<pow('+str(randomsize[23])+'*x1,2) ? lr :(pow(x[0]-x1*0.9,2)+pow(x[1]-x1*0.9,2))<pow('+str(randomsize[24])+'*x1,2) ? lr :h0','0','0'), degree=2,x1=x1,h0=h0,lr=lr)

    # Define trial and test functions within function space
    v, w,s = TestFunctions(VWS)
    hpm = Function(VWS)
    h,p,m, = split(hpm)
    # Initialize solution in our function space at t=0
    hpm_n=project(hpm_0, VWS)
    h_n, p_n,m_n = split(hpm_n)
    # Make eta a function
    elmts = VectorElement('Lagrange', mesh.ufl_cell(), 1)
    etaV=FunctionSpace(mesh,elmts)
    eta=Function(etaV)
    
    ## Formulate the variational problem
    # Write out weak form of the equations with all terms on one side
    #
    #gammel med feilfaktorF=h_n*v*dx -h*v*dx-dt*Q*h**(3/2)*dot(grad(v),eta)*dx-dt*h**3*(1/(12))*dot(grad(p),grad(v))*dx +p*w*dx +B*dot(grad(w),grad(m))*dx -1*w*(exp(-((lr-h)/(sigmaon*lr))**2)/(exp(-((lr-h)/(sigmaon*lr))**2)+taur))*(h-lr)*dx - 1*w*lr*(exp(-((1-h)/(sigmaon*1))**2)/(exp(-((1-h)/(sigmaon*1))**2)+taur))*(h-1)*dx  +m*s*dx +dot(grad(s),grad(h))*dx   #+z1*c1_n*dx -z1*c1*dx +dt*z1*dot(grad(p),grad(c1))*(h*l1/mu)*dx +dt*(c10-c1)*z1*((1/tau1on)*exp(((l1-h)/(sigmaon*l1))**2))*dx -dt*c1*z1*((1/tau1off)*exp(((l1-h)/(sigmaoff*l1))**2))*dx -dt*D1*dot(grad(z1),grad(c1))*dx   +z2*c2_n*dx -z2*c2*dx +dt*z2*dot(grad(p),grad(c2))*(h*l2/mu)*dx +dt*(c20-c2)*z2*((1/tau2on)*exp(((l2-h)/(sigmaon*l2))**2))*dx -dt*c2*z2*((1/tau2off)*exp(((l2-h)/(sigmaoff*l2))**2))*dx -dt*D2*dot(grad(z2),grad(c2))*dx     
    F=h_n*v*dx -h*v*dx-dt*h**3*dot(grad(p),grad(v))*dx +p*w*dx +B*dot(grad(w),grad(m))*dx -1*w*(exp(-((lr-h)/(sigmaon*lr))**2)/(exp(-((lr-h)/(sigmaon*lr))**2)+taur))*(h-lr)*dx - 1*w*lr*(exp(-((1-h)/(sigmaon*1))**2)/(exp(-((1-h)/(sigmaon*1))**2)+taur))*(h-1)*dx  +m*s*dx +dot(grad(s),grad(h))*dx   #+z1*c1_n*dx -z1*c1*dx +dt*z1*dot(grad(p),grad(c1))*(h*l1/mu)*dx +dt*(c10-c1)*z1*((1/tau1on)*exp(((l1-h)/(sigmaon*l1))**2))*dx -dt*c1*z1*((1/tau1off)*exp(((l1-h)/(sigmaoff*l1))**2))*dx -dt*D1*dot(grad(z1),grad(c1))*dx   +z2*c2_n*dx -z2*c2*dx +dt*z2*dot(grad(p),grad(c2))*(h*l2/mu)*dx +dt*(c20-c2)*z2*((1/tau2on)*exp(((l2-h)/(sigmaon*l2))**2))*dx -dt*c2*z2*((1/tau2off)*exp(((l2-h)/(sigmaoff*l2))**2))*dx -dt*D2*dot(grad(z2),grad(c2))*dx     

    J=derivative(F,hpm)

    problem=NonlinearVariationalProblem(F,hpm,J=J)
    solver=NonlinearVariationalSolver(problem)
    
    #% lag filer der data kan lagres
    t_series=open('t.txt','w')   # create and open text files in which to write down parameter values
    t_profiles=open('t_profil.txt','w')
    h_series=open('h.txt','w')
    xdmffile = XDMFFile('solution.xdmf') #create xdmf file for the solution
    xdmffile.parameters["flush_output"] = True
    profiles=np.zeros((nx+1,nx+1,nt+1))
    ts=np.zeros(nt)
    xs=np.zeros(nx+1)
    ys=np.zeros(nx+1)
    profts=np.zeros((nt+1))
    x=mesh.coordinates()
    xs=x[0:nx+1,0]
    ys=x[0:nx+1,0]
    nxd=nx
    xd=np.linspace(x0,x1,num=nxd)
    min=1000
    max=1000
    xmin=0
    for i1 in range(nxd-1):
        for i2 in range(nxd-1):
            val=hpm_n(xd[i1],xd[i2])[0]
            if val < min:
                min=val
                idmx=i1
                idmy=i2
    xmin=xd[idmx]
    ymin=xd[idmy]
    h_series.write(str(round(min,6)))       # write value
    h_series.write('\n') 


# =============================================================================
#     def stochastic(Number,deltax,dt):
#         mesh = UnitIntervalMesh(Number)        # lag mesh
#         elms = FiniteElement('Lagrange', mesh.ufl_cell(), 1)  #
#         U = FunctionSpace(mesh, elms)     # funksjonsrom,
#         u = TrialFunction(U)
#         v = TestFunction(U)
#         eta = Function(U)
#         stdevinv = deltax*np.sqrt(dt)
#         stdev=1/stdevinv
#         def update_random(random=eta):    # funksjon
#             random.vector().set_local(np.random.normal(0, stdev, random.vector().local_size()))
#             random.vector().apply('insert')
#         eq = inner(u, v)*dx - inner(eta, v)*dx
#         u = Function(U)
#         update_random()
#         solve(lhs(eq) == rhs(eq), u)
#         outfile = open('u.txt', 'w')
#         for i in range(len(mesh.coordinates())):
#             outfile.write('%f\t %f\n' %(mesh.coordinates()[i], u(mesh.coordinates()[i])))
#         outfile.close()
#         
# =============================================================================
    #% registrer metadata
    simpars=open('simparametere.txt','w')
    simpars.write('nx = '+str(nx))
    simpars.write('\n')
    simpars.write('x0 = '+str(x0))
    simpars.write('\n')
    simpars.write('x1 = '+str(x1))
    simpars.write('\n')
    simpars.write('dt = '+str(dt0))
    simpars.write('\n')
    simpars.write('Q = '+str(Q))
    simpars.write('\n')
    simpars.write('B = '+str(B))
    simpars.write('\n')
    simpars.write('h_0 = '+str(h0))
    simpars.write('\n')

    #% Initialisér løsning
    F0=h*v*dx-h_n*v*dx +p*w*dx-p_n*w*dx   +m*s*dx -m_n*s*dx 
    solve(F0==0,hpm)
    
    #% Loop i tid
    t=0
    profts[0]=t
    _h0,_p0, _m0= hpm.split()
    xdmffile.write(_h0,t)
    t_series.write(str(round(t,6)))
    t_series.write('\n')
    t_profiles.write(str(round(t,6)))
    t_profiles.write('\n')

    for ix in range(nx+1):
        for iy in range(nx+1):
            profiles[ix,iy,0]=h0


    itno_old =0
    i=0
    iprof=0
    imax=-1000
    vals = np.zeros(2)
    klokk0=0
    klokk1=0
    klokk2=0
    while t<t_max:
        i+=1
        print('\n   \n')
        print( 'Q = '+str(Q)+', h0 = '+str(h0)+' -- Sim nummer '+str(simnum))
       
        klokk2=klokk1
        klokk1=klokk0
        klokk0=time.time()
        klokkdelt=np.mean([klokk0-klokk1,klokk1-klokk2])
        irem=t_max/dt0-i
        trem=klokkdelt*irem
        if i>=3:
            print('Estimated time remaining = '+str(datetime.timedelta(seconds=klokkdelt*irem)))
        
        # update current timestep
        t+=float(dt)                       # Time for which we will be computing solution
        print('Timestep number: '+str(i))#+'  /'+str(Nt))
        print('t = '+str(round(float(t),7)))#+'  /'+str(T))
# =============================================================================
#         stochastic(len(eta.vector()),deltax,dt0)    # oppdater eta i tekstfil
#         
# =============================================================================
        #etaraw=np.loadtxt('u.txt') # last inn eta
        #etaraw=etaraw[:,1]          # ta bare andre kolonne

# =============================================================================
#         for ik in range(len(etaraw)-1):
#             #print(ik)
#             eta.vector()[ik] = etaraw[ik]         # sett vektorverdiene inn i funksjonen eta                         *********************
#             if (np.sqrt(eta.vector()[ik]**2-etaraw[ik]**2)) != 0:
#                 print('Fail to assign new values to eta')
# 
# =============================================================================

        # solve the variational problem to get new u at current timestep
        itno,conv =solver.solve()
        
        _h,_p, _m= hpm.split()
    
        hpm_n.assign(hpm)
        t_series.write(str(t))
        t_series.write('\n')
        

        if profilyn==1 and i/profilinterval==int(i/profilinterval):
            iprof+=1
            t_profiles.write(str(round(t,6)))
            t_profiles.write('\n')
            profts[iprof]=t
            for ix in range(nx+1):
                for iy in range(nx+1):
                    profiles[ix,iy,iprof]=_h(xs[ix],ys[iy])
            xdmffile.write(_h,t)
            
        min=1000
        for i1 in range(nxd-1):
            for i2 in range(nxd-1):
                val=hpm_n(xd[i1],xd[i2])[0]
                if val < min:
                    min=val
                    idmx=i1
                    idmy=i2
        h_series.write(str(round(min,6)))       # write value
        h_series.write('\n') 
    #%
    h_series.close()
    t_series.close()
    t_profiles.close()

    endtime=time.time()
    print('Elapsed time = '+str(datetime.timedelta(seconds=endtime-starttime)))
    simpars.write('Elapsed real time = '+str(datetime.timedelta(seconds=endtime-starttime)))
    simpars.write('\n')
    simpars.write('# of iterations = '+str(i))
    simpars.write('\n')
    simpars.write('Simulation end time = '+str(t))
    simpars.close()

    #%
    np.savez('profiles.npz',profiles=profiles,xs=xs,ys=ys,profts=profts)

def fasesep_prote1n_bend(Q,B,h0,l2,tau2off,sigmaon,x1,nx,profilinterval,profilyn,deltax,dt,t_max,simnum):
    starttime=time.time()

    Q=float(Q)
    
    B=float(B)
    tau2on=tau2off/3
    taur= tau2on/tau2off
    
    dt0=dt
    x0=0
    lr=1/3
    nt=int(t_max/(dt0*profilinterval))
    
    tau2on=float(tau2on)
    tau2off=float(tau2off)
    sigmaon=float(sigmaon)
    
    l2=float(l2)
    h0=float(h0)
    
    #% Lag mesh
    dt=Constant(dt)
    deltax=(x1-x0)/nx

    mesh = RectangleMesh( Point(x0,x0), Point(x1, x1), nx, nx, diagonal='right')
    #mesh = generate_mesh(domain)
    #plot(mesh)
    class Periodic_sides(SubDomain):
        def inside(self, x, on_boundary):
            return bool((near(x[0], 0) or near(x[1], 0)) and (not ((near(x[0],0) and near(x[1],x1)) or (near(x[0],x1) and near(x[1],0)))) and on_boundary)

        def map(self, x, y):
            if near(x[0],x1) and near(x[1],x1):
                y[0] = x[0] - x1
                y[1] = x[1] - x1
            elif near(x[0],x1):
                y[0] = x[0] - x1
                y[1] = x[1]
            elif near(x[1],x1):
                y[0] = x[0]
                y[1] = x[1] - x1
            else:
                y[0]=-1000
                y[1]=-1000  
        
        
    #% Define mixed function space
    EL = FiniteElement('P',triangle,1)
    element = MixedElement([EL,EL,EL])
    VWS = FunctionSpace(mesh,element, constrained_domain=Periodic_sides())
    # Define initial condition 
    hpm_0 = Expression(('h0','0','0'), degree=2,x1=x1,h0=h0)

    # Define trial and test functions within function space
    v, w,s = TestFunctions(VWS)
    hpm = Function(VWS)
    h,p,m, = split(hpm)
    # Initialize solution in our function space at t=0
    hpm_n=project(hpm_0, VWS)
    h_n, p_n,m_n = split(hpm_n)
    # Make eta a function
    elmts = VectorElement('Lagrange', mesh.ufl_cell(), 1)
    etaV=FunctionSpace(mesh,elmts)
    eta=Function(etaV)
    
    ## Formulate the variational problem
    # Write out weak form of the equations with all terms on one side
    #
    #gammel med feilfaktorF=h_n*v*dx -h*v*dx-dt*Q*h**(3/2)*dot(grad(v),eta)*dx-dt*h**3*(1/(12))*dot(grad(p),grad(v))*dx +p*w*dx +B*dot(grad(w),grad(m))*dx -1*w*(exp(-((lr-h)/(sigmaon*lr))**2)/(exp(-((lr-h)/(sigmaon*lr))**2)+taur))*(h-lr)*dx - 1*w*lr*(exp(-((1-h)/(sigmaon*1))**2)/(exp(-((1-h)/(sigmaon*1))**2)+taur))*(h-1)*dx  +m*s*dx +dot(grad(s),grad(h))*dx   #+z1*c1_n*dx -z1*c1*dx +dt*z1*dot(grad(p),grad(c1))*(h*l1/mu)*dx +dt*(c10-c1)*z1*((1/tau1on)*exp(((l1-h)/(sigmaon*l1))**2))*dx -dt*c1*z1*((1/tau1off)*exp(((l1-h)/(sigmaoff*l1))**2))*dx -dt*D1*dot(grad(z1),grad(c1))*dx   +z2*c2_n*dx -z2*c2*dx +dt*z2*dot(grad(p),grad(c2))*(h*l2/mu)*dx +dt*(c20-c2)*z2*((1/tau2on)*exp(((l2-h)/(sigmaon*l2))**2))*dx -dt*c2*z2*((1/tau2off)*exp(((l2-h)/(sigmaoff*l2))**2))*dx -dt*D2*dot(grad(z2),grad(c2))*dx     
    F=h_n*v*dx -h*v*dx-dt*Q*(h**(3/2))*dot(grad(v),eta)*dx-dt*(h**3)*dot(grad(p),grad(v))*dx +p*w*dx +B*dot(grad(w),grad(m))*dx - 1*w*1*(exp(-((1-h)/(sigmaon*1))**2)/(exp(-((1-h)/(sigmaon*1))**2)+taur))*(h-1)*dx  +m*s*dx +dot(grad(s),grad(h))*dx   #+z1*c1_n*dx -z1*c1*dx +dt*z1*dot(grad(p),grad(c1))*(h*l1/mu)*dx +dt*(c10-c1)*z1*((1/tau1on)*exp(((l1-h)/(sigmaon*l1))**2))*dx -dt*c1*z1*((1/tau1off)*exp(((l1-h)/(sigmaoff*l1))**2))*dx -dt*D1*dot(grad(z1),grad(c1))*dx   +z2*c2_n*dx -z2*c2*dx +dt*z2*dot(grad(p),grad(c2))*(h*l2/mu)*dx +dt*(c20-c2)*z2*((1/tau2on)*exp(((l2-h)/(sigmaon*l2))**2))*dx -dt*c2*z2*((1/tau2off)*exp(((l2-h)/(sigmaoff*l2))**2))*dx -dt*D2*dot(grad(z2),grad(c2))*dx     

    J=derivative(F,hpm)

    problem=NonlinearVariationalProblem(F,hpm,J=J)
    solver=NonlinearVariationalSolver(problem)
    
    #% lag filer der data kan lagres
    t_series=open('t.txt','w')   # create and open text files in which to write down parameter values
    t_profiles=open('t_profil.txt','w')
    h_series=open('h.txt','w')
    if profilyn==1:
        xdmffile = XDMFFile('solution.xdmf') #create xdmf file for the solution
        xdmffile.parameters["flush_output"] = True
    profiles=np.zeros((nx+1,nx+1,nt+1))
    ts=np.zeros(nt)
    xs=np.zeros(nx+1)
    ys=np.zeros(nx+1)
    profts=np.zeros((nt+1))
    x=mesh.coordinates()
    xs=x[0:nx+1,0]
    ys=x[0:nx+1,0]
    nxd=nx
    xd=np.linspace(x0,x1,num=nxd)
    min=1000
    max=1000
    xmin=0
    for i1 in range(nxd-1):
        for i2 in range(nxd-1):
            val=hpm_n(xd[i1],xd[i2])[0]
            if val < min:
                min=val
                idmx=i1
                idmy=i2
    xmin=xd[idmx]
    ymin=xd[idmy]
    h_series.write(str(round(min,6)))       # write value
    h_series.write('\n') 


    def stochastic(Number,deltax,dt):
        mesh = UnitIntervalMesh(Number)        # lag mesh
        elms = FiniteElement('Lagrange', mesh.ufl_cell(), 1)  #
        U = FunctionSpace(mesh, elms)     # funksjonsrom,
        u = TrialFunction(U)
        v = TestFunction(U)
        eta = Function(U)
        stdevinv = deltax*np.sqrt(dt)
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
        
    #% registrer metadata
    simpars=open('simparametere.txt','w')
    simpars.write('nx = '+str(nx))
    simpars.write('\n')
    simpars.write('x0 = '+str(x0))
    simpars.write('\n')
    simpars.write('x1 = '+str(x1))
    simpars.write('\n')
    simpars.write('dt = '+str(dt0))
    simpars.write('\n')
    simpars.write('Q = '+str(Q))
    simpars.write('\n')
    simpars.write('B = '+str(B))
    simpars.write('\n')
    simpars.write('h_0 = '+str(h0))
    simpars.write('\n')

    #% Initialisér løsning
    F0=h*v*dx-h_n*v*dx +p*w*dx-p_n*w*dx   +m*s*dx -m_n*s*dx 
    solve(F0==0,hpm)
    
    #% Loop i tid
    t=0
    profts[0]=t
    _h0,_p0, _m0= hpm.split()
    if profilyn==1:
        xdmffile.write(_h0,t)
    
    
    t_series.write(str(round(t,6)))
    t_series.write('\n')
    t_profiles.write(str(round(t,6)))
    t_profiles.write('\n')

    for ix in range(nx+1):
        for iy in range(nx+1):
            profiles[ix,iy,0]=h0


    itno_old =0
    i=0
    iprof=0
    imax=-1000
    vals = np.zeros(2)
    klokk0=0
    klokk1=0
    klokk2=0
    while t<t_max:
        i+=1
        print('\n   \n')
        print( 'Q = '+str(Q)+', h0 = '+str(h0)+' -- Sim nummer '+str(simnum))
        klokk2=klokk1
        klokk1=klokk0
        klokk0=time.time()
        klokkdelt=np.mean([klokk0-klokk1,klokk1-klokk2])
        irem=t_max/dt0-i
        trem=klokkdelt*irem
        if i>=3:
            print('Estimated time remaining = '+str(datetime.timedelta(seconds=klokkdelt*irem)))
        # update current timestep
        t+=float(dt)                       # Time for which we will be computing solution
        print('Iteration number: '+str(i))#+'  /'+str(Nt))
        print('t = '+str(round(float(t),7)))#+'  /'+str(T))
        stochastic(len(eta.vector()),deltax,dt0)    # oppdater eta i tekstfil
        
        etaraw=np.loadtxt('u.txt') # last inn eta
        etaraw=etaraw[:,1]          # ta bare andre kolonne

        for ik in range(len(etaraw)-1):
            #print(ik)
            eta.vector()[ik] = etaraw[ik]         # sett vektorverdiene inn i funksjonen eta                         *********************
            if (np.sqrt(eta.vector()[ik]**2-etaraw[ik]**2)) != 0:
                print('Fail to assign new values to eta')


        # solve the variational problem to get new u at current timestep
        itno,conv =solver.solve()
        
        _h,_p, _m= hpm.split()
    
        hpm_n.assign(hpm)
        t_series.write(str(t))
        t_series.write('\n')
        

        if  i/profilinterval==int(i/profilinterval) and t<=t_max:
            iprof+=1
            t_profiles.write(str(round(t,6)))
            t_profiles.write('\n')
            profts[iprof]=t
            for ix in range(nx+1):
                for iy in range(nx+1):
                    profiles[ix,iy,iprof]=_h(xs[ix],ys[iy])
            if profilyn==1:
                xdmffile.write(_h,t)
            
        min=1000
        for i1 in range(nxd-1):
            for i2 in range(nxd-1):
                val=hpm_n(xd[i1],xd[i2])[0]
                if val < min:
                    min=val
                    idmx=i1
                    idmy=i2
        h_series.write(str(round(min,6)))       # write value
        h_series.write('\n') 
        print('minimum height = '+str(min))
    #%
    h_series.close()
    t_series.close()
    t_profiles.close()

    endtime=time.time()
    print('Elapsed time = '+str(datetime.timedelta(seconds=endtime-starttime)))
    simpars.write('Elapsed real time = '+str(datetime.timedelta(seconds=endtime-starttime)))
    simpars.write('\n')
    simpars.write('# of iterations = '+str(i))
    simpars.write('\n')
    simpars.write('Simulation end time = '+str(t))
    simpars.close()

    #%
    np.savez('profiles.npz',profiles=profiles,xs=xs,ys=ys,profts=profts)
    # Get the current date and time
    now = datetime.datetime.now()

    # Create a datetime object representing the current date and time

    # Display a message indicating what is being printed
    print("Current date and time : ")

    # Print the current date and time in a specific format
    print(now.strftime("%Y-%m-%d %H:%M:%S"))

def fasesep_prote1n_bend_constantmob(Q,B,h0,l2,tau2off,sigmaon,x1,nx,profilinterval,profilyn,deltax,dt,t_max,simnum):
    starttime=time.time()

    Q=float(Q)
    
    B=float(B)
    tau2on=tau2off/3
    taur= tau2on/tau2off
    
    dt0=dt
    x0=0
    lr=1/3
    nt=int(t_max/(dt0*profilinterval))
    
    tau2on=float(tau2on)
    tau2off=float(tau2off)
    sigmaon=float(sigmaon)
    
    l2=float(l2)
    h0=float(h0)
    
    #% Lag mesh
    dt=Constant(dt)
    deltax=(x1-x0)/nx

    mesh = RectangleMesh( Point(x0,x0), Point(x1, x1), nx, nx, diagonal='right')
    #mesh = generate_mesh(domain)
    #plot(mesh)
    class Periodic_sides(SubDomain):
        def inside(self, x, on_boundary):
            return bool((near(x[0], 0) or near(x[1], 0)) and (not ((near(x[0],0) and near(x[1],x1)) or (near(x[0],x1) and near(x[1],0)))) and on_boundary)

        def map(self, x, y):
            if near(x[0],x1) and near(x[1],x1):
                y[0] = x[0] - x1
                y[1] = x[1] - x1
            elif near(x[0],x1):
                y[0] = x[0] - x1
                y[1] = x[1]
            elif near(x[1],x1):
                y[0] = x[0]
                y[1] = x[1] - x1
            else:
                y[0]=-1000
                y[1]=-1000  
        
        
    #% Define mixed function space
    EL = FiniteElement('P',triangle,1)
    element = MixedElement([EL,EL,EL])
    VWS = FunctionSpace(mesh,element, constrained_domain=Periodic_sides())
    # Define initial condition 
    hpm_0 = Expression(('h0','0','0'), degree=2,x1=x1,h0=h0)

    # Define trial and test functions within function space
    v, w,s = TestFunctions(VWS)
    hpm = Function(VWS)
    h,p,m, = split(hpm)
    # Initialize solution in our function space at t=0
    hpm_n=project(hpm_0, VWS)
    h_n, p_n,m_n = split(hpm_n)
    # Make eta a function
    elmts = VectorElement('Lagrange', mesh.ufl_cell(), 1)
    etaV=FunctionSpace(mesh,elmts)
    eta=Function(etaV)
    
    ## Formulate the variational problem
    # Write out weak form of the equations with all terms on one side
    #
    #gammel med feilfaktorF=h_n*v*dx -h*v*dx-dt*Q*h**(3/2)*dot(grad(v),eta)*dx-dt*h**3*(1/(12))*dot(grad(p),grad(v))*dx +p*w*dx +B*dot(grad(w),grad(m))*dx -1*w*(exp(-((lr-h)/(sigmaon*lr))**2)/(exp(-((lr-h)/(sigmaon*lr))**2)+taur))*(h-lr)*dx - 1*w*lr*(exp(-((1-h)/(sigmaon*1))**2)/(exp(-((1-h)/(sigmaon*1))**2)+taur))*(h-1)*dx  +m*s*dx +dot(grad(s),grad(h))*dx   #+z1*c1_n*dx -z1*c1*dx +dt*z1*dot(grad(p),grad(c1))*(h*l1/mu)*dx +dt*(c10-c1)*z1*((1/tau1on)*exp(((l1-h)/(sigmaon*l1))**2))*dx -dt*c1*z1*((1/tau1off)*exp(((l1-h)/(sigmaoff*l1))**2))*dx -dt*D1*dot(grad(z1),grad(c1))*dx   +z2*c2_n*dx -z2*c2*dx +dt*z2*dot(grad(p),grad(c2))*(h*l2/mu)*dx +dt*(c20-c2)*z2*((1/tau2on)*exp(((l2-h)/(sigmaon*l2))**2))*dx -dt*c2*z2*((1/tau2off)*exp(((l2-h)/(sigmaoff*l2))**2))*dx -dt*D2*dot(grad(z2),grad(c2))*dx     
    F=h_n*v*dx -h*v*dx-dt*Q*(h**(0/2))*dot(grad(v),eta)*dx-dt*(h**0)*dot(grad(p),grad(v))*dx +p*w*dx +B*dot(grad(w),grad(m))*dx - 1*w*1*(exp(-((1-h)/(sigmaon*1))**2)/(exp(-((1-h)/(sigmaon*1))**2)+taur))*(h-1)*dx  +m*s*dx +dot(grad(s),grad(h))*dx   #+z1*c1_n*dx -z1*c1*dx +dt*z1*dot(grad(p),grad(c1))*(h*l1/mu)*dx +dt*(c10-c1)*z1*((1/tau1on)*exp(((l1-h)/(sigmaon*l1))**2))*dx -dt*c1*z1*((1/tau1off)*exp(((l1-h)/(sigmaoff*l1))**2))*dx -dt*D1*dot(grad(z1),grad(c1))*dx   +z2*c2_n*dx -z2*c2*dx +dt*z2*dot(grad(p),grad(c2))*(h*l2/mu)*dx +dt*(c20-c2)*z2*((1/tau2on)*exp(((l2-h)/(sigmaon*l2))**2))*dx -dt*c2*z2*((1/tau2off)*exp(((l2-h)/(sigmaoff*l2))**2))*dx -dt*D2*dot(grad(z2),grad(c2))*dx     

    J=derivative(F,hpm)

    problem=NonlinearVariationalProblem(F,hpm,J=J)
    solver=NonlinearVariationalSolver(problem)
    
    #% lag filer der data kan lagres
    t_series=open('t.txt','w')   # create and open text files in which to write down parameter values
    t_profiles=open('t_profil.txt','w')
    h_series=open('h.txt','w')
    if profilyn==1:
        xdmffile = XDMFFile('solution.xdmf') #create xdmf file for the solution
        xdmffile.parameters["flush_output"] = True
    profiles=np.zeros((nx+1,nx+1,nt+1))
    ts=np.zeros(nt)
    xs=np.zeros(nx+1)
    ys=np.zeros(nx+1)
    profts=np.zeros((nt+1))
    x=mesh.coordinates()
    xs=x[0:nx+1,0]
    ys=x[0:nx+1,0]
    nxd=nx
    xd=np.linspace(x0,x1,num=nxd)
    min=1000
    max=1000
    xmin=0
    for i1 in range(nxd-1):
        for i2 in range(nxd-1):
            val=hpm_n(xd[i1],xd[i2])[0]
            if val < min:
                min=val
                idmx=i1
                idmy=i2
    xmin=xd[idmx]
    ymin=xd[idmy]
    h_series.write(str(round(min,6)))       # write value
    h_series.write('\n') 


    def stochastic(Number,deltax,dt):
        mesh = UnitIntervalMesh(Number)        # lag mesh
        elms = FiniteElement('Lagrange', mesh.ufl_cell(), 1)  #
        U = FunctionSpace(mesh, elms)     # funksjonsrom,
        u = TrialFunction(U)
        v = TestFunction(U)
        eta = Function(U)
        stdevinv = deltax*np.sqrt(dt)
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
        
    #% registrer metadata
    simpars=open('simparametere.txt','w')
    simpars.write('nx = '+str(nx))
    simpars.write('\n')
    simpars.write('x0 = '+str(x0))
    simpars.write('\n')
    simpars.write('x1 = '+str(x1))
    simpars.write('\n')
    simpars.write('dt = '+str(dt0))
    simpars.write('\n')
    simpars.write('Q = '+str(Q))
    simpars.write('\n')
    simpars.write('B = '+str(B))
    simpars.write('\n')
    simpars.write('h_0 = '+str(h0))
    simpars.write('\n')

    #% Initialisér løsning
    F0=h*v*dx-h_n*v*dx +p*w*dx-p_n*w*dx   +m*s*dx -m_n*s*dx 
    solve(F0==0,hpm)
    
    #% Loop i tid
    t=0
    profts[0]=t
    _h0,_p0, _m0= hpm.split()
    if profilyn==1:
        xdmffile.write(_h0,t)
    
    
    t_series.write(str(round(t,6)))
    t_series.write('\n')
    t_profiles.write(str(round(t,6)))
    t_profiles.write('\n')

    for ix in range(nx+1):
        for iy in range(nx+1):
            profiles[ix,iy,0]=h0


    itno_old =0
    i=0
    iprof=0
    imax=-1000
    vals = np.zeros(2)
    klokk0=0
    klokk1=0
    klokk2=0
    while t<t_max:
        i+=1
        print('\n   \n')
        print( 'Q = '+str(Q)+', h0 = '+str(h0)+' -- Sim nummer '+str(simnum))
        klokk2=klokk1
        klokk1=klokk0
        klokk0=time.time()
        klokkdelt=np.mean([klokk0-klokk1,klokk1-klokk2])
        irem=t_max/dt0-i
        trem=klokkdelt*irem
        if i>=3:
            print('Estimated time remaining = '+str(datetime.timedelta(seconds=klokkdelt*irem)))
        # update current timestep
        t+=float(dt)                       # Time for which we will be computing solution
        print('Iteration number: '+str(i))#+'  /'+str(Nt))
        print('t = '+str(round(float(t),7)))#+'  /'+str(T))
        stochastic(len(eta.vector()),deltax,dt0)    # oppdater eta i tekstfil
        
        etaraw=np.loadtxt('u.txt') # last inn eta
        etaraw=etaraw[:,1]          # ta bare andre kolonne

        for ik in range(len(etaraw)-1):
            #print(ik)
            eta.vector()[ik] = etaraw[ik]         # sett vektorverdiene inn i funksjonen eta                         *********************
            if (np.sqrt(eta.vector()[ik]**2-etaraw[ik]**2)) != 0:
                print('Fail to assign new values to eta')


        # solve the variational problem to get new u at current timestep
        itno,conv =solver.solve()
        
        _h,_p, _m= hpm.split()
    
        hpm_n.assign(hpm)
        t_series.write(str(t))
        t_series.write('\n')
        

        if  i/profilinterval==int(i/profilinterval) and t<=t_max:
            iprof+=1
            t_profiles.write(str(round(t,6)))
            t_profiles.write('\n')
            profts[iprof]=t
            for ix in range(nx+1):
                for iy in range(nx+1):
                    profiles[ix,iy,iprof]=_h(xs[ix],ys[iy])
            if profilyn==1:
                xdmffile.write(_h,t)
            
        min=1000
        for i1 in range(nxd-1):
            for i2 in range(nxd-1):
                val=hpm_n(xd[i1],xd[i2])[0]
                if val < min:
                    min=val
                    idmx=i1
                    idmy=i2
        h_series.write(str(round(min,6)))       # write value
        h_series.write('\n') 
        print('minimum height = '+str(min))
    #%
    h_series.close()
    t_series.close()
    t_profiles.close()

    endtime=time.time()
    print('Elapsed time = '+str(datetime.timedelta(seconds=endtime-starttime)))
    simpars.write('Elapsed real time = '+str(datetime.timedelta(seconds=endtime-starttime)))
    simpars.write('\n')
    simpars.write('# of iterations = '+str(i))
    simpars.write('\n')
    simpars.write('Simulation end time = '+str(t))
    simpars.close()

    #%
    np.savez('profiles.npz',profiles=profiles,xs=xs,ys=ys,profts=profts)

def fasesep_prote1n_bend_Darcy(Q,B,h0,l2,tau2off,sigmaon,x1,nx,profilinterval,profilyn,deltax,dt,t_max,simnum):
    starttime=time.time()

    Q=float(Q)
    
    B=float(B)
    tau2on=tau2off/3
    taur= tau2on/tau2off
    
    dt0=dt
    x0=0
    lr=1/3
    nt=int(t_max/(dt0*profilinterval))
    
    tau2on=float(tau2on)
    tau2off=float(tau2off)
    sigmaon=float(sigmaon)
    
    l2=float(l2)
    h0=float(h0)
    
    #% Lag mesh
    dt=Constant(dt)
    deltax=(x1-x0)/nx

    mesh = RectangleMesh( Point(x0,x0), Point(x1, x1), nx, nx, diagonal='right')
    #mesh = generate_mesh(domain)
    #plot(mesh)
    class Periodic_sides(SubDomain):
        def inside(self, x, on_boundary):
            return bool((near(x[0], 0) or near(x[1], 0)) and (not ((near(x[0],0) and near(x[1],x1)) or (near(x[0],x1) and near(x[1],0)))) and on_boundary)

        def map(self, x, y):
            if near(x[0],x1) and near(x[1],x1):
                y[0] = x[0] - x1
                y[1] = x[1] - x1
            elif near(x[0],x1):
                y[0] = x[0] - x1
                y[1] = x[1]
            elif near(x[1],x1):
                y[0] = x[0]
                y[1] = x[1] - x1
            else:
                y[0]=-1000
                y[1]=-1000  
        
        
    #% Define mixed function space
    EL = FiniteElement('P',triangle,1)
    element = MixedElement([EL,EL,EL])
    VWS = FunctionSpace(mesh,element, constrained_domain=Periodic_sides())
    # Define initial condition 
    hpm_0 = Expression(('h0','0','0'), degree=2,x1=x1,h0=h0)

    # Define trial and test functions within function space
    v, w,s = TestFunctions(VWS)
    hpm = Function(VWS)
    h,p,m, = split(hpm)
    # Initialize solution in our function space at t=0
    hpm_n=project(hpm_0, VWS)
    h_n, p_n,m_n = split(hpm_n)
    # Make eta a function
    elmts = VectorElement('Lagrange', mesh.ufl_cell(), 1)
    etaV=FunctionSpace(mesh,elmts)
    eta=Function(etaV)
    
    ## Formulate the variational problem
    # Write out weak form of the equations with all terms on one side
    #
    #gammel med feilfaktorF=h_n*v*dx -h*v*dx-dt*Q*h**(3/2)*dot(grad(v),eta)*dx-dt*h**3*(1/(12))*dot(grad(p),grad(v))*dx +p*w*dx +B*dot(grad(w),grad(m))*dx -1*w*(exp(-((lr-h)/(sigmaon*lr))**2)/(exp(-((lr-h)/(sigmaon*lr))**2)+taur))*(h-lr)*dx - 1*w*lr*(exp(-((1-h)/(sigmaon*1))**2)/(exp(-((1-h)/(sigmaon*1))**2)+taur))*(h-1)*dx  +m*s*dx +dot(grad(s),grad(h))*dx   #+z1*c1_n*dx -z1*c1*dx +dt*z1*dot(grad(p),grad(c1))*(h*l1/mu)*dx +dt*(c10-c1)*z1*((1/tau1on)*exp(((l1-h)/(sigmaon*l1))**2))*dx -dt*c1*z1*((1/tau1off)*exp(((l1-h)/(sigmaoff*l1))**2))*dx -dt*D1*dot(grad(z1),grad(c1))*dx   +z2*c2_n*dx -z2*c2*dx +dt*z2*dot(grad(p),grad(c2))*(h*l2/mu)*dx +dt*(c20-c2)*z2*((1/tau2on)*exp(((l2-h)/(sigmaon*l2))**2))*dx -dt*c2*z2*((1/tau2off)*exp(((l2-h)/(sigmaoff*l2))**2))*dx -dt*D2*dot(grad(z2),grad(c2))*dx     
    F=h_n*v*dx -h*v*dx-dt*Q*(h**(1/2))*dot(grad(v),eta)*dx-dt*(h**1)*dot(grad(p),grad(v))*dx +p*w*dx +B*dot(grad(w),grad(m))*dx - 1*w*1*(exp(-((1-h)/(sigmaon*1))**2)/(exp(-((1-h)/(sigmaon*1))**2)+taur))*(h-1)*dx  +m*s*dx +dot(grad(s),grad(h))*dx   #+z1*c1_n*dx -z1*c1*dx +dt*z1*dot(grad(p),grad(c1))*(h*l1/mu)*dx +dt*(c10-c1)*z1*((1/tau1on)*exp(((l1-h)/(sigmaon*l1))**2))*dx -dt*c1*z1*((1/tau1off)*exp(((l1-h)/(sigmaoff*l1))**2))*dx -dt*D1*dot(grad(z1),grad(c1))*dx   +z2*c2_n*dx -z2*c2*dx +dt*z2*dot(grad(p),grad(c2))*(h*l2/mu)*dx +dt*(c20-c2)*z2*((1/tau2on)*exp(((l2-h)/(sigmaon*l2))**2))*dx -dt*c2*z2*((1/tau2off)*exp(((l2-h)/(sigmaoff*l2))**2))*dx -dt*D2*dot(grad(z2),grad(c2))*dx     

    J=derivative(F,hpm)

    problem=NonlinearVariationalProblem(F,hpm,J=J)
    solver=NonlinearVariationalSolver(problem)
    
    #% lag filer der data kan lagres
    t_series=open('t.txt','w')   # create and open text files in which to write down parameter values
    t_profiles=open('t_profil.txt','w')
    h_series=open('h.txt','w')
    if profilyn==1:
        xdmffile = XDMFFile('solution.xdmf') #create xdmf file for the solution
        xdmffile.parameters["flush_output"] = True
    profiles=np.zeros((nx+1,nx+1,nt+1))
    ts=np.zeros(nt)
    xs=np.zeros(nx+1)
    ys=np.zeros(nx+1)
    profts=np.zeros((nt+1))
    x=mesh.coordinates()
    xs=x[0:nx+1,0]
    ys=x[0:nx+1,0]
    nxd=nx
    xd=np.linspace(x0,x1,num=nxd)
    min=1000
    max=1000
    xmin=0
    for i1 in range(nxd-1):
        for i2 in range(nxd-1):
            val=hpm_n(xd[i1],xd[i2])[0]
            if val < min:
                min=val
                idmx=i1
                idmy=i2
    xmin=xd[idmx]
    ymin=xd[idmy]
    h_series.write(str(round(min,6)))       # write value
    h_series.write('\n') 


    def stochastic(Number,deltax,dt):
        mesh = UnitIntervalMesh(Number)        # lag mesh
        elms = FiniteElement('Lagrange', mesh.ufl_cell(), 1)  #
        U = FunctionSpace(mesh, elms)     # funksjonsrom,
        u = TrialFunction(U)
        v = TestFunction(U)
        eta = Function(U)
        stdevinv = deltax*np.sqrt(dt)
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
        
    #% registrer metadata
    simpars=open('simparametere.txt','w')
    simpars.write('nx = '+str(nx))
    simpars.write('\n')
    simpars.write('x0 = '+str(x0))
    simpars.write('\n')
    simpars.write('x1 = '+str(x1))
    simpars.write('\n')
    simpars.write('dt = '+str(dt0))
    simpars.write('\n')
    simpars.write('Q = '+str(Q))
    simpars.write('\n')
    simpars.write('B = '+str(B))
    simpars.write('\n')
    simpars.write('h_0 = '+str(h0))
    simpars.write('\n')

    #% Initialisér løsning
    F0=h*v*dx-h_n*v*dx +p*w*dx-p_n*w*dx   +m*s*dx -m_n*s*dx 
    solve(F0==0,hpm)
    
    #% Loop i tid
    t=0
    profts[0]=t
    _h0,_p0, _m0= hpm.split()
    if profilyn==1:
        xdmffile.write(_h0,t)
    
    
    t_series.write(str(round(t,6)))
    t_series.write('\n')
    t_profiles.write(str(round(t,6)))
    t_profiles.write('\n')

    for ix in range(nx+1):
        for iy in range(nx+1):
            profiles[ix,iy,0]=h0


    itno_old =0
    i=0
    iprof=0
    imax=-1000
    vals = np.zeros(2)
    klokk0=0
    klokk1=0
    klokk2=0
    while t<t_max:
        i+=1
        print('\n   \n')
        print( 'Q = '+str(Q)+', h0 = '+str(h0)+' -- Sim nummer '+str(simnum))
        klokk2=klokk1
        klokk1=klokk0
        klokk0=time.time()
        klokkdelt=np.mean([klokk0-klokk1,klokk1-klokk2])
        irem=t_max/dt0-i
        trem=klokkdelt*irem
        if i>=3:
            print('Estimated time remaining = '+str(datetime.timedelta(seconds=klokkdelt*irem)))
        # update current timestep
        t+=float(dt)                       # Time for which we will be computing solution
        print('Iteration number: '+str(i))#+'  /'+str(Nt))
        print('t = '+str(round(float(t),7)))#+'  /'+str(T))
        stochastic(len(eta.vector()),deltax,dt0)    # oppdater eta i tekstfil
        
        etaraw=np.loadtxt('u.txt') # last inn eta
        etaraw=etaraw[:,1]          # ta bare andre kolonne

        for ik in range(len(etaraw)-1):
            #print(ik)
            eta.vector()[ik] = etaraw[ik]         # sett vektorverdiene inn i funksjonen eta                         *********************
            if (np.sqrt(eta.vector()[ik]**2-etaraw[ik]**2)) != 0:
                print('Fail to assign new values to eta')


        # solve the variational problem to get new u at current timestep
        itno,conv =solver.solve()
        
        _h,_p, _m= hpm.split()
    
        hpm_n.assign(hpm)
        t_series.write(str(t))
        t_series.write('\n')
        

        if  i/profilinterval==int(i/profilinterval) and t<=t_max:
            iprof+=1
            t_profiles.write(str(round(t,6)))
            t_profiles.write('\n')
            profts[iprof]=t
            for ix in range(nx+1):
                for iy in range(nx+1):
                    profiles[ix,iy,iprof]=_h(xs[ix],ys[iy])
            if profilyn==1:
                xdmffile.write(_h,t)
            
        min=1000
        for i1 in range(nxd-1):
            for i2 in range(nxd-1):
                val=hpm_n(xd[i1],xd[i2])[0]
                if val < min:
                    min=val
                    idmx=i1
                    idmy=i2
        h_series.write(str(round(min,6)))       # write value
        h_series.write('\n') 
        print('minimum height = '+str(min))
    #%
    h_series.close()
    t_series.close()
    t_profiles.close()

    endtime=time.time()
    print('Elapsed time = '+str(datetime.timedelta(seconds=endtime-starttime)))
    simpars.write('Elapsed real time = '+str(datetime.timedelta(seconds=endtime-starttime)))
    simpars.write('\n')
    simpars.write('# of iterations = '+str(i))
    simpars.write('\n')
    simpars.write('Simulation end time = '+str(t))
    simpars.close()

    #%
    np.savez('profiles.npz',profiles=profiles,xs=xs,ys=ys,profts=profts)
    # Get the current date and time
    now = datetime.datetime.now()

    # Create a datetime object representing the current date and time

    # Display a message indicating what is being printed
    print("Current date and time : ")

    # Print the current date and time in a specific format
    print(now.strftime("%Y-%m-%d %H:%M:%S"))



def fasesep_prote1n_begge(Q,K,gamma,h0,l2,tau2off,sigmaon,x1,nx,profilinterval,profilinterval2,profilyn,deltax,dt,t_max,simnum):
    starttime=time.time()

    Q=float(Q)
    
    K=float(K)
    gamma=float(gamma)
    tau2on=tau2off/3
    taur= tau2on/tau2off
    
    dt0=dt
    x0=0
    lr=1/3
    nt=int(t_max/(dt0*profilinterval))
    
    tau2on=float(tau2on)
    tau2off=float(tau2off)
    sigmaon=float(sigmaon)
    
    l2=float(l2)
    h0=float(h0)
    
    #% Lag mesh
    dt=Constant(dt)
    deltax=(x1-x0)/nx

    mesh = RectangleMesh( Point(x0,x0), Point(x1, x1), nx, nx, diagonal='right')
    #mesh = generate_mesh(domain)
    #plot(mesh)
    class Periodic_sides(SubDomain):
        def inside(self, x, on_boundary):
            return bool((near(x[0], 0) or near(x[1], 0)) and (not ((near(x[0],0) and near(x[1],x1)) or (near(x[0],x1) and near(x[1],0)))) and on_boundary)

        def map(self, x, y):
            if near(x[0],x1) and near(x[1],x1):
                y[0] = x[0] - x1
                y[1] = x[1] - x1
            elif near(x[0],x1):
                y[0] = x[0] - x1
                y[1] = x[1]
            elif near(x[1],x1):
                y[0] = x[0]
                y[1] = x[1] - x1
            else:
                y[0]=-1000
                y[1]=-1000  
        
        
    #% Define mixed function space
    EL = FiniteElement('P',triangle,1)
    element = MixedElement([EL,EL,EL])
    VWS = FunctionSpace(mesh,element, constrained_domain=Periodic_sides())
    # Define initial condition 
    hpm_0 = Expression(('h0','0','0'), degree=2,x1=x1,h0=h0)

    # Define trial and test functions within function space
    v, w,s = TestFunctions(VWS)
    hpm = Function(VWS)
    h,p,m, = split(hpm)
    # Initialize solution in our function space at t=0
    hpm_n=project(hpm_0, VWS)
    h_n, p_n,m_n = split(hpm_n)
    # Make eta a function
    elmts = VectorElement('Lagrange', mesh.ufl_cell(), 1)
    etaV=FunctionSpace(mesh,elmts)
    eta=Function(etaV)
    
    ## Formulate the variational problem
    # Write out weak form of the equations with all terms on one side
    #
    #gammel med feilfaktorF=h_n*v*dx -h*v*dx-dt*Q*h**(3/2)*dot(grad(v),eta)*dx-dt*h**3*(1/(12))*dot(grad(p),grad(v))*dx +p*w*dx +B*dot(grad(w),grad(m))*dx -1*w*(exp(-((lr-h)/(sigmaon*lr))**2)/(exp(-((lr-h)/(sigmaon*lr))**2)+taur))*(h-lr)*dx - 1*w*lr*(exp(-((1-h)/(sigmaon*1))**2)/(exp(-((1-h)/(sigmaon*1))**2)+taur))*(h-1)*dx  +m*s*dx +dot(grad(s),grad(h))*dx   #+z1*c1_n*dx -z1*c1*dx +dt*z1*dot(grad(p),grad(c1))*(h*l1/mu)*dx +dt*(c10-c1)*z1*((1/tau1on)*exp(((l1-h)/(sigmaon*l1))**2))*dx -dt*c1*z1*((1/tau1off)*exp(((l1-h)/(sigmaoff*l1))**2))*dx -dt*D1*dot(grad(z1),grad(c1))*dx   +z2*c2_n*dx -z2*c2*dx +dt*z2*dot(grad(p),grad(c2))*(h*l2/mu)*dx +dt*(c20-c2)*z2*((1/tau2on)*exp(((l2-h)/(sigmaon*l2))**2))*dx -dt*c2*z2*((1/tau2off)*exp(((l2-h)/(sigmaoff*l2))**2))*dx -dt*D2*dot(grad(z2),grad(c2))*dx     
    F=h_n*v*dx -h*v*dx-dt*Q*h**(3/2)*dot(grad(v),eta)*dx-dt*h**3*dot(grad(p),grad(v))*dx +p*w*dx-gamma*dot(grad(w),grad(h))*dx +1*dot(grad(w),grad(m))*dx - K*w*1*(exp(-((1-h)/(sigmaon*1))**2)/(exp(-((1-h)/(sigmaon*1))**2)+taur))*(h-1)*dx  +m*s*dx +dot(grad(s),grad(h))*dx   #+z1*c1_n*dx -z1*c1*dx +dt*z1*dot(grad(p),grad(c1))*(h*l1/mu)*dx +dt*(c10-c1)*z1*((1/tau1on)*exp(((l1-h)/(sigmaon*l1))**2))*dx -dt*c1*z1*((1/tau1off)*exp(((l1-h)/(sigmaoff*l1))**2))*dx -dt*D1*dot(grad(z1),grad(c1))*dx   +z2*c2_n*dx -z2*c2*dx +dt*z2*dot(grad(p),grad(c2))*(h*l2/mu)*dx +dt*(c20-c2)*z2*((1/tau2on)*exp(((l2-h)/(sigmaon*l2))**2))*dx -dt*c2*z2*((1/tau2off)*exp(((l2-h)/(sigmaoff*l2))**2))*dx -dt*D2*dot(grad(z2),grad(c2))*dx     

    J=derivative(F,hpm)

    problem=NonlinearVariationalProblem(F,hpm,J=J)
    solver=NonlinearVariationalSolver(problem)
    
    #% lag filer der data kan lagres
    t_series=open('t.txt','w')   # create and open text files in which to write down parameter values
    t_profiles=open('t_profil.txt','w')
    h_series=open('h.txt','w')
    if profilyn==1:
        xdmffile = XDMFFile('solution.xdmf') #create xdmf file for the solution
        xdmffile.parameters["flush_output"] = True
    profiles=np.zeros((nx+1,nx+1,nt+1))
    ts=np.zeros(nt)
    xs=np.zeros(nx+1)
    ys=np.zeros(nx+1)
    profts=np.zeros((nt+1))
    x=mesh.coordinates()
    xs=x[0:nx+1,0]
    ys=x[0:nx+1,0]
    nxd=nx
    xd=np.linspace(x0,x1,num=nxd)
    min=1000
    max=1000
    xmin=0
    for i1 in range(nxd-1):
        for i2 in range(nxd-1):
            val=hpm_n(xd[i1],xd[i2])[0]
            if val < min:
                min=val
                idmx=i1
                idmy=i2
    xmin=xd[idmx]
    ymin=xd[idmy]
    h_series.write(str(round(min,6)))       # write value
    h_series.write('\n') 


    def stochastic(Number,deltax,dt):
        mesh = UnitIntervalMesh(Number)        # lag mesh
        elms = FiniteElement('Lagrange', mesh.ufl_cell(), 1)  #
        U = FunctionSpace(mesh, elms)     # funksjonsrom,
        u = TrialFunction(U)
        v = TestFunction(U)
        eta = Function(U)
        stdevinv = deltax*np.sqrt(dt)
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
        
    #% registrer metadata
    simpars=open('simparametere.txt','w')
    simpars.write('nx = '+str(nx))
    simpars.write('\n')
    simpars.write('x0 = '+str(x0))
    simpars.write('\n')
    simpars.write('x1 = '+str(x1))
    simpars.write('\n')
    simpars.write('dt = '+str(dt0))
    simpars.write('\n')
    simpars.write('Q = '+str(Q))
    simpars.write('\n')
    simpars.write('K = '+str(K))
    simpars.write('\n')
    simpars.write('h_0 = '+str(h0))
    simpars.write('\n')

    #% Initialisér løsning
    F0=h*v*dx-h_n*v*dx +p*w*dx-p_n*w*dx   +m*s*dx -m_n*s*dx 
    solve(F0==0,hpm)
    
    #% Loop i tid
    t=0
    profts[0]=t
    _h0,_p0, _m0= hpm.split()
    if profilyn==1:
        xdmffile.write(_h0,t)
    
    
    t_series.write(str(round(t,6)))
    t_series.write('\n')
    t_profiles.write(str(round(t,6)))
    t_profiles.write('\n')

    for ix in range(nx+1):
        for iy in range(nx+1):
            profiles[ix,iy,0]=h0


    itno_old =0
    i=0
    iprof=0
    imax=-1000
    vals = np.zeros(2)
    klokk0=0
    klokk1=0
    klokk2=0
    while t<t_max:
        i+=1
        print('\n   \n')
        print( 'Q = '+str(Q)+', h0 = '+str(h0)+' -- Sim nummer '+str(simnum))
        klokk2=klokk1
        klokk1=klokk0
        klokk0=time.time()
        klokkdelt=np.mean([klokk0-klokk1,klokk1-klokk2])
        irem=t_max/dt0-i
        trem=klokkdelt*irem
        if t>2000:
            profilinterval=profilinterval2
        if i>=3:
            print('Estimated time remaining = '+str(datetime.timedelta(seconds=klokkdelt*irem)))
        # update current timestep
        t+=float(dt)                       # Time for which we will be computing solution
        print('Iteration number: '+str(i))#+'  /'+str(Nt))
        print('t = '+str(round(float(t),7)))#+'  /'+str(T))
        stochastic(len(eta.vector()),deltax,dt0)    # oppdater eta i tekstfil
        
        etaraw=np.loadtxt('u.txt') # last inn eta
        etaraw=etaraw[:,1]          # ta bare andre kolonne

        for ik in range(len(etaraw)-1):
            #print(ik)
            eta.vector()[ik] = etaraw[ik]         # sett vektorverdiene inn i funksjonen eta                         *********************
            if (np.sqrt(eta.vector()[ik]**2-etaraw[ik]**2)) != 0:
                print('Fail to assign new values to eta')


        # solve the variational problem to get new u at current timestep
        itno,conv =solver.solve()
        
        _h,_p, _m= hpm.split()
    
        hpm_n.assign(hpm)
        t_series.write(str(t))
        t_series.write('\n')
        

        if  i/profilinterval==int(i/profilinterval) and t<=t_max:
            iprof+=1
            t_profiles.write(str(round(t,6)))
            t_profiles.write('\n')
            profts[iprof]=t
            for ix in range(nx+1):
                for iy in range(nx+1):
                    profiles[ix,iy,iprof]=_h(xs[ix],ys[iy])
            if profilyn==1:
                xdmffile.write(_h,t)
            
        min=1000
        for i1 in range(nxd-1):
            for i2 in range(nxd-1):
                val=hpm_n(xd[i1],xd[i2])[0]
                if val < min:
                    min=val
                    idmx=i1
                    idmy=i2
        h_series.write(str(round(min,6)))       # write value
        h_series.write('\n') 
        print('minimum height = '+str(min))
    #%
    h_series.close()
    t_series.close()
    t_profiles.close()

    endtime=time.time()
    print('Elapsed time = '+str(datetime.timedelta(seconds=endtime-starttime)))
    simpars.write('Elapsed real time = '+str(datetime.timedelta(seconds=endtime-starttime)))
    simpars.write('\n')
    simpars.write('# of iterations = '+str(i))
    simpars.write('\n')
    simpars.write('Simulation end time = '+str(t))
    simpars.close()

    #%
    np.savez('profiles.npz',profiles=profiles,xs=xs,ys=ys,profts=profts)

def fasesep_prote1n_int(Q,B,h0,l2,tau2off,sigmaon,x1,nx,profilinterval,profilyn,deltax,dt,t_max,simnum):
    starttime=time.time()

    Q=float(Q)
    
    B=float(B)
    tau2on=tau2off/3
    taur= tau2on/tau2off
    
    dt0=dt
    x0=0
    lr=1/3
    nt=int(t_max/(dt0*profilinterval))
    
    tau2on=float(tau2on)
    tau2off=float(tau2off)
    sigmaon=float(sigmaon)
    
    l2=float(l2)
    h0=float(h0)
    
    #% Lag mesh
    dt=Constant(dt)
    deltax=(x1-x0)/nx

    mesh = RectangleMesh( Point(x0,x0), Point(x1, x1), nx, nx, diagonal='right')
    #mesh = generate_mesh(domain)
    #plot(mesh)
    class Periodic_sides(SubDomain):
        def inside(self, x, on_boundary):
            return bool((near(x[0], 0) or near(x[1], 0)) and (not ((near(x[0],0) and near(x[1],x1)) or (near(x[0],x1) and near(x[1],0)))) and on_boundary)

        def map(self, x, y):
            if near(x[0],x1) and near(x[1],x1):
                y[0] = x[0] - x1
                y[1] = x[1] - x1
            elif near(x[0],x1):
                y[0] = x[0] - x1
                y[1] = x[1]
            elif near(x[1],x1):
                y[0] = x[0]
                y[1] = x[1] - x1
            else:
                y[0]=-1000
                y[1]=-1000  
        
        
    #% Define mixed function space
    EL = FiniteElement('P',triangle,1)
    element = MixedElement([EL,EL])
    VW = FunctionSpace(mesh,element, constrained_domain=Periodic_sides())
    # Define initial condition 
    hp_0 = Expression(('h0','0'), degree=2,x1=x1,h0=h0)

    # Define trial and test functions within function space
    v, w = TestFunctions(VW)
    hp = Function(VW)
    h,p, = split(hp)
    # Initialize solution in our function space at t=0
    hp_n=project(hp_0, VW)
    h_n, p_n = split(hp_n)
    # Make eta a function
    elmts = VectorElement('Lagrange', mesh.ufl_cell(), 1)
    etaV=FunctionSpace(mesh,elmts)
    eta=Function(etaV)
    
    ## Formulate the variational problem
    # Write out weak form of the equations with all terms on one side
    #
    #gammel med feilfaktorF=h_n*v*dx -h*v*dx-dt*Q*h**(3/2)*dot(grad(v),eta)*dx-dt*h**3*(1/(12))*dot(grad(p),grad(v))*dx +p*w*dx +B*dot(grad(w),grad(m))*dx -1*w*(exp(-((lr-h)/(sigmaon*lr))**2)/(exp(-((lr-h)/(sigmaon*lr))**2)+taur))*(h-lr)*dx - 1*w*lr*(exp(-((1-h)/(sigmaon*1))**2)/(exp(-((1-h)/(sigmaon*1))**2)+taur))*(h-1)*dx  +m*s*dx +dot(grad(s),grad(h))*dx   #+z1*c1_n*dx -z1*c1*dx +dt*z1*dot(grad(p),grad(c1))*(h*l1/mu)*dx +dt*(c10-c1)*z1*((1/tau1on)*exp(((l1-h)/(sigmaon*l1))**2))*dx -dt*c1*z1*((1/tau1off)*exp(((l1-h)/(sigmaoff*l1))**2))*dx -dt*D1*dot(grad(z1),grad(c1))*dx   +z2*c2_n*dx -z2*c2*dx +dt*z2*dot(grad(p),grad(c2))*(h*l2/mu)*dx +dt*(c20-c2)*z2*((1/tau2on)*exp(((l2-h)/(sigmaon*l2))**2))*dx -dt*c2*z2*((1/tau2off)*exp(((l2-h)/(sigmaoff*l2))**2))*dx -dt*D2*dot(grad(z2),grad(c2))*dx     
    #F=h*v*dx-h_n*v*dx-dt*Q*h**(3/2)*dot(grad(v),eta)*dx-dt*h**3*dot(grad(p),grad(v))*dx +p*w*dx  +dot(grad(w),grad(h))*dx+ 1*w*1*(exp(-((1-h)/(sigmaon*1))**2)/(exp(-((1-h)/(sigmaon*1))**2)+taur))*(h-1)*dx     #+z1*c1_n*dx -z1*c1*dx +dt*z1*dot(grad(p),grad(c1))*(h*l1/mu)*dx +dt*(c10-c1)*z1*((1/tau1on)*exp(((l1-h)/(sigmaon*l1))**2))*dx -dt*c1*z1*((1/tau1off)*exp(((l1-h)/(sigmaoff*l1))**2))*dx -dt*D1*dot(grad(z1),grad(c1))*dx   +z2*c2_n*dx -z2*c2*dx +dt*z2*dot(grad(p),grad(c2))*(h*l2/mu)*dx +dt*(c20-c2)*z2*((1/tau2on)*exp(((l2-h)/(sigmaon*l2))**2))*dx -dt*c2*z2*((1/tau2off)*exp(((l2-h)/(sigmaoff*l2))**2))*dx -dt*D2*dot(grad(z2),grad(c2))*dx     
    #F=h*v*dx -h_n*v*dx-dt*Q*h**(3/2)*dot(grad(v),eta)*dx-dt*h**3*dot(grad(p),grad(v))*dx +p*w*dx +dot(grad(w),grad(h))*dx + 1*w*1*(exp(-((1-h)/(sigmaon*1))**2)/(exp(-((1-h)/(sigmaon*1))**2)+taur))*(h-1)*dx  #+m*s*dx +dot(grad(s),grad(h))*dx
    #F=h*v*dx-h_n*v*dx-dt*(1/2)*Q*v.dx(0)*(h**(3/2)+h_n**(3/2))*eta*dx-dt*(1/2)*(h**3+h_n**3)*dot(grad(p),grad(v))*dx +p*w*dx+dot(grad(w),grad(h))*dx-1*w*1*(exp(-((1-h)/(sigmaon*1))**2)/(exp(-((1-h)/(sigmaon*1))**2)+taur))*(h-1)*dx
    F=h_n*v*dx -h*v*dx-dt*Q*h**(3/2)*dot(grad(v),eta)*dx-dt*h**3*dot(grad(p),grad(v))*dx +p*w*dx -dot(grad(w),grad(h))*dx - 1*w*1*(exp(-((1-h)/(sigmaon*1))**2)/(exp(-((1-h)/(sigmaon*1))**2)+taur))*(h-1)*dx#  +m*s*dx +dot(grad(s),grad(h))*dx   #+z1*c1_n*dx -z1*c1*dx +dt*z1*dot(grad(p),grad(c1))*(h*l1/mu)*dx +dt*(c10-c1)*z1*((1/tau1on)*exp(((l1-h)/(sigmaon*l1))**2))*dx -dt*c1*z1*((1/tau1off)*exp(((l1-h)/(sigmaoff*l1))**2))*dx -dt*D1*dot(grad(z1),grad(c1))*dx   +z2*c2_n*dx -z2*c2*dx +dt*z2*dot(grad(p),grad(c2))*(h*l2/mu)*dx +dt*(c20-c2)*z2*((1/tau2on)*exp(((l2-h)/(sigmaon*l2))**2))*dx -dt*c2*z2*((1/tau2off)*exp(((l2-h)/(sigmaoff*l2))**2))*dx -dt*D2*dot(grad(z2),grad(c2))*dx     

    J=derivative(F,hp)

    problem=NonlinearVariationalProblem(F,hp,J=J)
    solver=NonlinearVariationalSolver(problem)
    
    #% lag filer der data kan lagres
    t_series=open('t.txt','w')   # create and open text files in which to write down parameter values
    t_profiles=open('t_profil.txt','w')
    h_series=open('h.txt','w')
    xdmffile = XDMFFile('solution.xdmf') #create xdmf file for the solution
    xdmffile.parameters["flush_output"] = True
    profiles=np.zeros((nx+1,nx+1,nt+1))
    ts=np.zeros(nt)
    xs=np.zeros(nx+1)
    ys=np.zeros(nx+1)
    profts=np.zeros((nt+1))
    x=mesh.coordinates()
    xs=x[0:nx+1,0]
    ys=x[0:nx+1,0]
    nxd=nx
    xd=np.linspace(x0,x1,num=nxd)
    min=1000
    max=1000
    xmin=0
    for i1 in range(nxd-1):
        for i2 in range(nxd-1):
            val=hp_n(xd[i1],xd[i2])[0]
            if val < min:
                min=val
                idmx=i1
                idmy=i2
    xmin=xd[idmx]
    ymin=xd[idmy]
    h_series.write(str(round(min,6)))       # write value
    h_series.write('\n') 


    def stochastic(Number,deltax,dt):
        mesh = UnitIntervalMesh(Number)        # lag mesh
        elms = FiniteElement('Lagrange', mesh.ufl_cell(), 1)  #
        U = FunctionSpace(mesh, elms)     # funksjonsrom,
        u = TrialFunction(U)
        v = TestFunction(U)
        eta = Function(U)
        stdevinv = deltax*np.sqrt(dt)
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
        
    #% registrer metadata
    simpars=open('simparametere.txt','w')
    simpars.write('nx = '+str(nx))
    simpars.write('\n')
    simpars.write('x0 = '+str(x0))
    simpars.write('\n')
    simpars.write('x1 = '+str(x1))
    simpars.write('\n')
    simpars.write('dt = '+str(dt0))
    simpars.write('\n')
    simpars.write('Q = '+str(Q))
    simpars.write('\n')
    simpars.write('B = '+str(B))
    simpars.write('\n')
    simpars.write('h_0 = '+str(h0))
    simpars.write('\n')

    #% Initialisér løsning
    F0=h*v*dx-h_n*v*dx +p*w*dx-p_n*w*dx   
    solve(F0==0,hp)
    
    #% Loop i tid
    t=0
    profts[0]=t
    _h0,_p0= hp.split()
    xdmffile.write(_h0,t)
    t_series.write(str(round(t,6)))
    t_series.write('\n')
    t_profiles.write(str(round(t,6)))
    t_profiles.write('\n')

    for ix in range(nx+1):
        for iy in range(nx+1):
            profiles[ix,iy,0]=h0


    itno_old =0
    i=0
    iprof=0
    imax=-1000
    vals = np.zeros(2)
    klokk0=0
    klokk1=0
    klokk2=0
    while t<t_max:
        i+=1
        print('\n   \n')
        print( 'Q = '+str(Q)+', h0 = '+str(h0)+' -- Sim nummer '+str(simnum))
        klokk2=klokk1
        klokk1=klokk0
        klokk0=time.time()
        klokkdelt=np.mean([klokk0-klokk1,klokk1-klokk2])
        irem=t_max/dt0-i
        trem=klokkdelt*irem
        if i>=3:
            print('Estimated time remaining = '+str(datetime.timedelta(seconds=klokkdelt*irem)))
# update current timestep
        t+=float(dt)                       # Time for which we will be computing solution
        print('Iteration number: '+str(i))#+'  /'+str(Nt))
        print('t = '+str(round(float(t),7)))#+'  /'+str(T))
        stochastic(len(eta.vector()),deltax,dt0)    # oppdater eta i tekstfil
        
        etaraw=np.loadtxt('u.txt') # last inn eta
        etaraw=etaraw[:,1]          # ta bare andre kolonne

        for ik in range(len(etaraw)-1):
            #print(ik)
            eta.vector()[ik] = etaraw[ik]         # sett vektorverdiene inn i funksjonen eta                         *********************
            if (np.sqrt(eta.vector()[ik]**2-etaraw[ik]**2)) != 0:
                print('Fail to assign new values to eta')


        # solve the variational problem to get new u at current timestep
        itno,conv =solver.solve()
        
        _h,_p= hp.split()
    
        hp_n.assign(hp)
        t_series.write(str(t))
        t_series.write('\n')
        

        if profilyn==1 and i/profilinterval==int(i/profilinterval) and t<=t_max:
            iprof+=1
            t_profiles.write(str(round(t,6)))
            t_profiles.write('\n')
            profts[iprof]=t
            for ix in range(nx+1):
                for iy in range(nx+1):
                    profiles[ix,iy,iprof]=_h(xs[ix],ys[iy])
            xdmffile.write(_h,t)
            
        min=1000
        for i1 in range(nxd-1):
            for i2 in range(nxd-1):
                val=hp_n(xd[i1],xd[i2])[0]
                if val < min:
                    min=val
                    idmx=i1
                    idmy=i2
        h_series.write(str(round(min,6)))       # write value
        h_series.write('\n') 
        print('minimum height = '+str(min))
    #%
    h_series.close()
    t_series.close()
    t_profiles.close()

    endtime=time.time()
    print('Elapsed time = '+str(datetime.timedelta(seconds=endtime-starttime)))
    simpars.write('Elapsed real time = '+str(datetime.timedelta(seconds=endtime-starttime)))
    simpars.write('\n')
    simpars.write('# of iterations = '+str(i))
    simpars.write('\n')
    simpars.write('Simulation end time = '+str(t))
    simpars.close()

    #%
    np.savez('profiles.npz',profiles=profiles,xs=xs,ys=ys,profts=profts)
    # Get the current date and time
    now = datetime.datetime.now()

    # Create a datetime object representing the current date and time

    # Display a message indicating what is being printed
    print("Current date and time : ")

    # Print the current date and time in a specific format
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
def fasesep_prote1n_int_Darcy(Q,B,h0,l2,tau2off,sigmaon,x1,nx,profilinterval,profilyn,deltax,dt,t_max,simnum):
    starttime=time.time()

    Q=float(Q)
    
    B=float(B)
    tau2on=tau2off/3
    taur= tau2on/tau2off
    
    dt0=dt
    x0=0
    lr=1/3
    nt=int(t_max/(dt0*profilinterval))
    
    tau2on=float(tau2on)
    tau2off=float(tau2off)
    sigmaon=float(sigmaon)
    
    l2=float(l2)
    h0=float(h0)
    
    #% Lag mesh
    dt=Constant(dt)
    deltax=(x1-x0)/nx

    mesh = RectangleMesh( Point(x0,x0), Point(x1, x1), nx, nx, diagonal='right')
    #mesh = generate_mesh(domain)
    #plot(mesh)
    class Periodic_sides(SubDomain):
        def inside(self, x, on_boundary):
            return bool((near(x[0], 0) or near(x[1], 0)) and (not ((near(x[0],0) and near(x[1],x1)) or (near(x[0],x1) and near(x[1],0)))) and on_boundary)

        def map(self, x, y):
            if near(x[0],x1) and near(x[1],x1):
                y[0] = x[0] - x1
                y[1] = x[1] - x1
            elif near(x[0],x1):
                y[0] = x[0] - x1
                y[1] = x[1]
            elif near(x[1],x1):
                y[0] = x[0]
                y[1] = x[1] - x1
            else:
                y[0]=-1000
                y[1]=-1000  
        
        
    #% Define mixed function space
    EL = FiniteElement('P',triangle,1)
    element = MixedElement([EL,EL])
    VW = FunctionSpace(mesh,element, constrained_domain=Periodic_sides())
    # Define initial condition 
    hp_0 = Expression(('h0','0'), degree=2,x1=x1,h0=h0)

    # Define trial and test functions within function space
    v, w = TestFunctions(VW)
    hp = Function(VW)
    h,p, = split(hp)
    # Initialize solution in our function space at t=0
    hp_n=project(hp_0, VW)
    h_n, p_n = split(hp_n)
    # Make eta a function
    elmts = VectorElement('Lagrange', mesh.ufl_cell(), 1)
    etaV=FunctionSpace(mesh,elmts)
    eta=Function(etaV)
    
    ## Formulate the variational problem
    # Write out weak form of the equations with all terms on one side
    #
    #gammel med feilfaktorF=h_n*v*dx -h*v*dx-dt*Q*h**(3/2)*dot(grad(v),eta)*dx-dt*h**3*(1/(12))*dot(grad(p),grad(v))*dx +p*w*dx +B*dot(grad(w),grad(m))*dx -1*w*(exp(-((lr-h)/(sigmaon*lr))**2)/(exp(-((lr-h)/(sigmaon*lr))**2)+taur))*(h-lr)*dx - 1*w*lr*(exp(-((1-h)/(sigmaon*1))**2)/(exp(-((1-h)/(sigmaon*1))**2)+taur))*(h-1)*dx  +m*s*dx +dot(grad(s),grad(h))*dx   #+z1*c1_n*dx -z1*c1*dx +dt*z1*dot(grad(p),grad(c1))*(h*l1/mu)*dx +dt*(c10-c1)*z1*((1/tau1on)*exp(((l1-h)/(sigmaon*l1))**2))*dx -dt*c1*z1*((1/tau1off)*exp(((l1-h)/(sigmaoff*l1))**2))*dx -dt*D1*dot(grad(z1),grad(c1))*dx   +z2*c2_n*dx -z2*c2*dx +dt*z2*dot(grad(p),grad(c2))*(h*l2/mu)*dx +dt*(c20-c2)*z2*((1/tau2on)*exp(((l2-h)/(sigmaon*l2))**2))*dx -dt*c2*z2*((1/tau2off)*exp(((l2-h)/(sigmaoff*l2))**2))*dx -dt*D2*dot(grad(z2),grad(c2))*dx     
    #F=h*v*dx-h_n*v*dx-dt*Q*h**(3/2)*dot(grad(v),eta)*dx-dt*h**3*dot(grad(p),grad(v))*dx +p*w*dx  +dot(grad(w),grad(h))*dx+ 1*w*1*(exp(-((1-h)/(sigmaon*1))**2)/(exp(-((1-h)/(sigmaon*1))**2)+taur))*(h-1)*dx     #+z1*c1_n*dx -z1*c1*dx +dt*z1*dot(grad(p),grad(c1))*(h*l1/mu)*dx +dt*(c10-c1)*z1*((1/tau1on)*exp(((l1-h)/(sigmaon*l1))**2))*dx -dt*c1*z1*((1/tau1off)*exp(((l1-h)/(sigmaoff*l1))**2))*dx -dt*D1*dot(grad(z1),grad(c1))*dx   +z2*c2_n*dx -z2*c2*dx +dt*z2*dot(grad(p),grad(c2))*(h*l2/mu)*dx +dt*(c20-c2)*z2*((1/tau2on)*exp(((l2-h)/(sigmaon*l2))**2))*dx -dt*c2*z2*((1/tau2off)*exp(((l2-h)/(sigmaoff*l2))**2))*dx -dt*D2*dot(grad(z2),grad(c2))*dx     
    #F=h*v*dx -h_n*v*dx-dt*Q*h**(3/2)*dot(grad(v),eta)*dx-dt*h**3*dot(grad(p),grad(v))*dx +p*w*dx +dot(grad(w),grad(h))*dx + 1*w*1*(exp(-((1-h)/(sigmaon*1))**2)/(exp(-((1-h)/(sigmaon*1))**2)+taur))*(h-1)*dx  #+m*s*dx +dot(grad(s),grad(h))*dx
    #F=h*v*dx-h_n*v*dx-dt*(1/2)*Q*v.dx(0)*(h**(3/2)+h_n**(3/2))*eta*dx-dt*(1/2)*(h**3+h_n**3)*dot(grad(p),grad(v))*dx +p*w*dx+dot(grad(w),grad(h))*dx-1*w*1*(exp(-((1-h)/(sigmaon*1))**2)/(exp(-((1-h)/(sigmaon*1))**2)+taur))*(h-1)*dx
    F=h_n*v*dx -h*v*dx-dt*Q*h**(3/2)*dot(grad(v),eta)*dx-dt*h*dot(grad(p),grad(v))*dx +p*w*dx -dot(grad(w),grad(h))*dx - 1*w*1*(exp(-((1-h)/(sigmaon*1))**2)/(exp(-((1-h)/(sigmaon*1))**2)+taur))*(h-1)*dx#  +m*s*dx +dot(grad(s),grad(h))*dx   #+z1*c1_n*dx -z1*c1*dx +dt*z1*dot(grad(p),grad(c1))*(h*l1/mu)*dx +dt*(c10-c1)*z1*((1/tau1on)*exp(((l1-h)/(sigmaon*l1))**2))*dx -dt*c1*z1*((1/tau1off)*exp(((l1-h)/(sigmaoff*l1))**2))*dx -dt*D1*dot(grad(z1),grad(c1))*dx   +z2*c2_n*dx -z2*c2*dx +dt*z2*dot(grad(p),grad(c2))*(h*l2/mu)*dx +dt*(c20-c2)*z2*((1/tau2on)*exp(((l2-h)/(sigmaon*l2))**2))*dx -dt*c2*z2*((1/tau2off)*exp(((l2-h)/(sigmaoff*l2))**2))*dx -dt*D2*dot(grad(z2),grad(c2))*dx     

    J=derivative(F,hp)

    problem=NonlinearVariationalProblem(F,hp,J=J)
    solver=NonlinearVariationalSolver(problem)
    
    #% lag filer der data kan lagres
    t_series=open('t.txt','w')   # create and open text files in which to write down parameter values
    t_profiles=open('t_profil.txt','w')
    h_series=open('h.txt','w')
    xdmffile = XDMFFile('solution.xdmf') #create xdmf file for the solution
    xdmffile.parameters["flush_output"] = True
    profiles=np.zeros((nx+1,nx+1,nt+1))
    ts=np.zeros(nt)
    xs=np.zeros(nx+1)
    ys=np.zeros(nx+1)
    profts=np.zeros((nt+1))
    x=mesh.coordinates()
    xs=x[0:nx+1,0]
    ys=x[0:nx+1,0]
    nxd=nx
    xd=np.linspace(x0,x1,num=nxd)
    min=1000
    max=1000
    xmin=0
    for i1 in range(nxd-1):
        for i2 in range(nxd-1):
            val=hp_n(xd[i1],xd[i2])[0]
            if val < min:
                min=val
                idmx=i1
                idmy=i2
    xmin=xd[idmx]
    ymin=xd[idmy]
    h_series.write(str(round(min,6)))       # write value
    h_series.write('\n') 


    def stochastic(Number,deltax,dt):
        mesh = UnitIntervalMesh(Number)        # lag mesh
        elms = FiniteElement('Lagrange', mesh.ufl_cell(), 1)  #
        U = FunctionSpace(mesh, elms)     # funksjonsrom,
        u = TrialFunction(U)
        v = TestFunction(U)
        eta = Function(U)
        stdevinv = deltax*np.sqrt(dt)
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
        
    #% registrer metadata
    simpars=open('simparametere.txt','w')
    simpars.write('nx = '+str(nx))
    simpars.write('\n')
    simpars.write('x0 = '+str(x0))
    simpars.write('\n')
    simpars.write('x1 = '+str(x1))
    simpars.write('\n')
    simpars.write('dt = '+str(dt0))
    simpars.write('\n')
    simpars.write('Q = '+str(Q))
    simpars.write('\n')
    simpars.write('B = '+str(B))
    simpars.write('\n')
    simpars.write('h_0 = '+str(h0))
    simpars.write('\n')

    #% Initialisér løsning
    F0=h*v*dx-h_n*v*dx +p*w*dx-p_n*w*dx   
    solve(F0==0,hp)
    
    #% Loop i tid
    t=0
    profts[0]=t
    _h0,_p0= hp.split()
    xdmffile.write(_h0,t)
    t_series.write(str(round(t,6)))
    t_series.write('\n')
    t_profiles.write(str(round(t,6)))
    t_profiles.write('\n')

    for ix in range(nx+1):
        for iy in range(nx+1):
            profiles[ix,iy,0]=h0


    itno_old =0
    i=0
    iprof=0
    imax=-1000
    vals = np.zeros(2)
    klokk0=0
    klokk1=0
    klokk2=0
    while t<t_max:
        i+=1
        print('\n   \n')
        print( 'Q = '+str(Q)+', h0 = '+str(h0)+' -- Sim nummer '+str(simnum))
        klokk2=klokk1
        klokk1=klokk0
        klokk0=time.time()
        klokkdelt=np.mean([klokk0-klokk1,klokk1-klokk2])
        irem=t_max/dt0-i
        trem=klokkdelt*irem
        if i>=3:
            print('Estimated time remaining = '+str(datetime.timedelta(seconds=klokkdelt*irem)))
# update current timestep
        t+=float(dt)                       # Time for which we will be computing solution
        print('Iteration number: '+str(i))#+'  /'+str(Nt))
        print('t = '+str(round(float(t),7)))#+'  /'+str(T))
        stochastic(len(eta.vector()),deltax,dt0)    # oppdater eta i tekstfil
        
        etaraw=np.loadtxt('u.txt') # last inn eta
        etaraw=etaraw[:,1]          # ta bare andre kolonne

        for ik in range(len(etaraw)-1):
            #print(ik)
            eta.vector()[ik] = etaraw[ik]         # sett vektorverdiene inn i funksjonen eta                         *********************
            if (np.sqrt(eta.vector()[ik]**2-etaraw[ik]**2)) != 0:
                print('Fail to assign new values to eta')


        # solve the variational problem to get new u at current timestep
        itno,conv =solver.solve()
        
        _h,_p= hp.split()
    
        hp_n.assign(hp)
        t_series.write(str(t))
        t_series.write('\n')
        

        if profilyn==1 and i/profilinterval==int(i/profilinterval) and t<=t_max:
            iprof+=1
            t_profiles.write(str(round(t,6)))
            t_profiles.write('\n')
            profts[iprof]=t
            for ix in range(nx+1):
                for iy in range(nx+1):
                    profiles[ix,iy,iprof]=_h(xs[ix],ys[iy])
            xdmffile.write(_h,t)
            
        min=1000
        for i1 in range(nxd-1):
            for i2 in range(nxd-1):
                val=hp_n(xd[i1],xd[i2])[0]
                if val < min:
                    min=val
                    idmx=i1
                    idmy=i2
        h_series.write(str(round(min,6)))       # write value
        h_series.write('\n') 
        print('minimum height = '+str(min))
    #%
    h_series.close()
    t_series.close()
    t_profiles.close()

    endtime=time.time()
    print('Elapsed time = '+str(datetime.timedelta(seconds=endtime-starttime)))
    simpars.write('Elapsed real time = '+str(datetime.timedelta(seconds=endtime-starttime)))
    simpars.write('\n')
    simpars.write('# of iterations = '+str(i))
    simpars.write('\n')
    simpars.write('Simulation end time = '+str(t))
    simpars.close()

    #%
    np.savez('profiles.npz',profiles=profiles,xs=xs,ys=ys,profts=profts)
    # Get the current date and time
    now = datetime.datetime.now()

    # Create a datetime object representing the current date and time

    # Display a message indicating what is being printed
    print("Current date and time : ")

    # Print the current date and time in a specific format
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
def fasesep_prote1n_int_constantmob(Q,B,h0,l2,tau2off,sigmaon,x1,nx,profilinterval,profilyn,deltax,dt,t_max,simnum):
    starttime=time.time()

    Q=float(Q)
    
    B=float(B)
    tau2on=tau2off/3
    taur= tau2on/tau2off
    
    dt0=dt
    x0=0
    lr=1/3
    nt=int(t_max/(dt0*profilinterval))
    
    tau2on=float(tau2on)
    tau2off=float(tau2off)
    sigmaon=float(sigmaon)
    
    l2=float(l2)
    h0=float(h0)
    
    #% Lag mesh
    dt=Constant(dt)
    deltax=(x1-x0)/nx

    mesh = RectangleMesh( Point(x0,x0), Point(x1, x1), nx, nx, diagonal='right')
    #mesh = generate_mesh(domain)
    #plot(mesh)
    class Periodic_sides(SubDomain):
        def inside(self, x, on_boundary):
            return bool((near(x[0], 0) or near(x[1], 0)) and (not ((near(x[0],0) and near(x[1],x1)) or (near(x[0],x1) and near(x[1],0)))) and on_boundary)

        def map(self, x, y):
            if near(x[0],x1) and near(x[1],x1):
                y[0] = x[0] - x1
                y[1] = x[1] - x1
            elif near(x[0],x1):
                y[0] = x[0] - x1
                y[1] = x[1]
            elif near(x[1],x1):
                y[0] = x[0]
                y[1] = x[1] - x1
            else:
                y[0]=-1000
                y[1]=-1000  
        
        
    #% Define mixed function space
    EL = FiniteElement('P',triangle,1)
    element = MixedElement([EL,EL])
    VW = FunctionSpace(mesh,element, constrained_domain=Periodic_sides())
    # Define initial condition 
    hp_0 = Expression(('h0','0'), degree=2,x1=x1,h0=h0)

    # Define trial and test functions within function space
    v, w = TestFunctions(VW)
    hp = Function(VW)
    h,p, = split(hp)
    # Initialize solution in our function space at t=0
    hp_n=project(hp_0, VW)
    h_n, p_n = split(hp_n)
    # Make eta a function
    elmts = VectorElement('Lagrange', mesh.ufl_cell(), 1)
    etaV=FunctionSpace(mesh,elmts)
    eta=Function(etaV)
    
    ## Formulate the variational problem
    # Write out weak form of the equations with all terms on one side
    #
    #gammel med feilfaktorF=h_n*v*dx -h*v*dx-dt*Q*h**(3/2)*dot(grad(v),eta)*dx-dt*h**3*(1/(12))*dot(grad(p),grad(v))*dx +p*w*dx +B*dot(grad(w),grad(m))*dx -1*w*(exp(-((lr-h)/(sigmaon*lr))**2)/(exp(-((lr-h)/(sigmaon*lr))**2)+taur))*(h-lr)*dx - 1*w*lr*(exp(-((1-h)/(sigmaon*1))**2)/(exp(-((1-h)/(sigmaon*1))**2)+taur))*(h-1)*dx  +m*s*dx +dot(grad(s),grad(h))*dx   #+z1*c1_n*dx -z1*c1*dx +dt*z1*dot(grad(p),grad(c1))*(h*l1/mu)*dx +dt*(c10-c1)*z1*((1/tau1on)*exp(((l1-h)/(sigmaon*l1))**2))*dx -dt*c1*z1*((1/tau1off)*exp(((l1-h)/(sigmaoff*l1))**2))*dx -dt*D1*dot(grad(z1),grad(c1))*dx   +z2*c2_n*dx -z2*c2*dx +dt*z2*dot(grad(p),grad(c2))*(h*l2/mu)*dx +dt*(c20-c2)*z2*((1/tau2on)*exp(((l2-h)/(sigmaon*l2))**2))*dx -dt*c2*z2*((1/tau2off)*exp(((l2-h)/(sigmaoff*l2))**2))*dx -dt*D2*dot(grad(z2),grad(c2))*dx     
    #F=h*v*dx-h_n*v*dx-dt*Q*h**(3/2)*dot(grad(v),eta)*dx-dt*h**3*dot(grad(p),grad(v))*dx +p*w*dx  +dot(grad(w),grad(h))*dx+ 1*w*1*(exp(-((1-h)/(sigmaon*1))**2)/(exp(-((1-h)/(sigmaon*1))**2)+taur))*(h-1)*dx     #+z1*c1_n*dx -z1*c1*dx +dt*z1*dot(grad(p),grad(c1))*(h*l1/mu)*dx +dt*(c10-c1)*z1*((1/tau1on)*exp(((l1-h)/(sigmaon*l1))**2))*dx -dt*c1*z1*((1/tau1off)*exp(((l1-h)/(sigmaoff*l1))**2))*dx -dt*D1*dot(grad(z1),grad(c1))*dx   +z2*c2_n*dx -z2*c2*dx +dt*z2*dot(grad(p),grad(c2))*(h*l2/mu)*dx +dt*(c20-c2)*z2*((1/tau2on)*exp(((l2-h)/(sigmaon*l2))**2))*dx -dt*c2*z2*((1/tau2off)*exp(((l2-h)/(sigmaoff*l2))**2))*dx -dt*D2*dot(grad(z2),grad(c2))*dx     
    #F=h*v*dx -h_n*v*dx-dt*Q*h**(3/2)*dot(grad(v),eta)*dx-dt*h**3*dot(grad(p),grad(v))*dx +p*w*dx +dot(grad(w),grad(h))*dx + 1*w*1*(exp(-((1-h)/(sigmaon*1))**2)/(exp(-((1-h)/(sigmaon*1))**2)+taur))*(h-1)*dx  #+m*s*dx +dot(grad(s),grad(h))*dx
    #F=h*v*dx-h_n*v*dx-dt*(1/2)*Q*v.dx(0)*(h**(3/2)+h_n**(3/2))*eta*dx-dt*(1/2)*(h**3+h_n**3)*dot(grad(p),grad(v))*dx +p*w*dx+dot(grad(w),grad(h))*dx-1*w*1*(exp(-((1-h)/(sigmaon*1))**2)/(exp(-((1-h)/(sigmaon*1))**2)+taur))*(h-1)*dx
    F=h_n*v*dx -h*v*dx-dt*Q*dot(grad(v),eta)*dx-dt*dot(grad(p),grad(v))*dx +p*w*dx -dot(grad(w),grad(h))*dx - 1*w*1*(exp(-((1-h)/(sigmaon*1))**2)/(exp(-((1-h)/(sigmaon*1))**2)+taur))*(h-1)*dx#  +m*s*dx +dot(grad(s),grad(h))*dx   #+z1*c1_n*dx -z1*c1*dx +dt*z1*dot(grad(p),grad(c1))*(h*l1/mu)*dx +dt*(c10-c1)*z1*((1/tau1on)*exp(((l1-h)/(sigmaon*l1))**2))*dx -dt*c1*z1*((1/tau1off)*exp(((l1-h)/(sigmaoff*l1))**2))*dx -dt*D1*dot(grad(z1),grad(c1))*dx   +z2*c2_n*dx -z2*c2*dx +dt*z2*dot(grad(p),grad(c2))*(h*l2/mu)*dx +dt*(c20-c2)*z2*((1/tau2on)*exp(((l2-h)/(sigmaon*l2))**2))*dx -dt*c2*z2*((1/tau2off)*exp(((l2-h)/(sigmaoff*l2))**2))*dx -dt*D2*dot(grad(z2),grad(c2))*dx     

    J=derivative(F,hp)

    problem=NonlinearVariationalProblem(F,hp,J=J)
    solver=NonlinearVariationalSolver(problem)
    
    #% lag filer der data kan lagres
    t_series=open('t.txt','w')   # create and open text files in which to write down parameter values
    t_profiles=open('t_profil.txt','w')
    h_series=open('h.txt','w')
    xdmffile = XDMFFile('solution.xdmf') #create xdmf file for the solution
    xdmffile.parameters["flush_output"] = True
    profiles=np.zeros((nx+1,nx+1,nt+1))
    ts=np.zeros(nt)
    xs=np.zeros(nx+1)
    ys=np.zeros(nx+1)
    profts=np.zeros((nt+1))
    x=mesh.coordinates()
    xs=x[0:nx+1,0]
    ys=x[0:nx+1,0]
    nxd=nx
    xd=np.linspace(x0,x1,num=nxd)
    min=1000
    max=1000
    xmin=0
    for i1 in range(nxd-1):
        for i2 in range(nxd-1):
            val=hp_n(xd[i1],xd[i2])[0]
            if val < min:
                min=val
                idmx=i1
                idmy=i2
    xmin=xd[idmx]
    ymin=xd[idmy]
    h_series.write(str(round(min,6)))       # write value
    h_series.write('\n') 


    def stochastic(Number,deltax,dt):
        mesh = UnitIntervalMesh(Number)        # lag mesh
        elms = FiniteElement('Lagrange', mesh.ufl_cell(), 1)  #
        U = FunctionSpace(mesh, elms)     # funksjonsrom,
        u = TrialFunction(U)
        v = TestFunction(U)
        eta = Function(U)
        stdevinv = deltax*np.sqrt(dt)
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
        
    #% registrer metadata
    simpars=open('simparametere.txt','w')
    simpars.write('nx = '+str(nx))
    simpars.write('\n')
    simpars.write('x0 = '+str(x0))
    simpars.write('\n')
    simpars.write('x1 = '+str(x1))
    simpars.write('\n')
    simpars.write('dt = '+str(dt0))
    simpars.write('\n')
    simpars.write('Q = '+str(Q))
    simpars.write('\n')
    simpars.write('B = '+str(B))
    simpars.write('\n')
    simpars.write('h_0 = '+str(h0))
    simpars.write('\n')

    #% Initialisér løsning
    F0=h*v*dx-h_n*v*dx +p*w*dx-p_n*w*dx   
    solve(F0==0,hp)
    
    #% Loop i tid
    t=0
    profts[0]=t
    _h0,_p0= hp.split()
    xdmffile.write(_h0,t)
    t_series.write(str(round(t,6)))
    t_series.write('\n')
    t_profiles.write(str(round(t,6)))
    t_profiles.write('\n')

    for ix in range(nx+1):
        for iy in range(nx+1):
            profiles[ix,iy,0]=h0


    itno_old =0
    i=0
    iprof=0
    imax=-1000
    vals = np.zeros(2)
    klokk0=0
    klokk1=0
    klokk2=0
    while t<t_max:
        i+=1
        print('\n   \n')
        print( 'Q = '+str(Q)+', h0 = '+str(h0)+' -- Sim nummer '+str(simnum))
        klokk2=klokk1
        klokk1=klokk0
        klokk0=time.time()
        klokkdelt=np.mean([klokk0-klokk1,klokk1-klokk2])
        irem=t_max/dt0-i
        trem=klokkdelt*irem
        if i>=3:
            print('Estimated time remaining = '+str(datetime.timedelta(seconds=klokkdelt*irem)))
# update current timestep
        t+=float(dt)                       # Time for which we will be computing solution
        print('Iteration number: '+str(i))#+'  /'+str(Nt))
        print('t = '+str(round(float(t),7)))#+'  /'+str(T))
        stochastic(len(eta.vector()),deltax,dt0)    # oppdater eta i tekstfil
        
        etaraw=np.loadtxt('u.txt') # last inn eta
        etaraw=etaraw[:,1]          # ta bare andre kolonne

        for ik in range(len(etaraw)-1):
            #print(ik)
            eta.vector()[ik] = etaraw[ik]         # sett vektorverdiene inn i funksjonen eta                         *********************
            if (np.sqrt(eta.vector()[ik]**2-etaraw[ik]**2)) != 0:
                print('Fail to assign new values to eta')


        # solve the variational problem to get new u at current timestep
        itno,conv =solver.solve()
        
        _h,_p= hp.split()
    
        hp_n.assign(hp)
        t_series.write(str(t))
        t_series.write('\n')
        

        if profilyn==1 and i/profilinterval==int(i/profilinterval) and t<=t_max:
            iprof+=1
            t_profiles.write(str(round(t,6)))
            t_profiles.write('\n')
            profts[iprof]=t
            for ix in range(nx+1):
                for iy in range(nx+1):
                    profiles[ix,iy,iprof]=_h(xs[ix],ys[iy])
            xdmffile.write(_h,t)
            
        min=1000
        for i1 in range(nxd-1):
            for i2 in range(nxd-1):
                val=hp_n(xd[i1],xd[i2])[0]
                if val < min:
                    min=val
                    idmx=i1
                    idmy=i2
        h_series.write(str(round(min,6)))       # write value
        h_series.write('\n') 
        print('minimum height = '+str(min))
    #%
    h_series.close()
    t_series.close()
    t_profiles.close()

    endtime=time.time()
    print('Elapsed time = '+str(datetime.timedelta(seconds=endtime-starttime)))
    simpars.write('Elapsed real time = '+str(datetime.timedelta(seconds=endtime-starttime)))
    simpars.write('\n')
    simpars.write('# of iterations = '+str(i))
    simpars.write('\n')
    simpars.write('Simulation end time = '+str(t))
    simpars.close()

    #%
    np.savez('profiles.npz',profiles=profiles,xs=xs,ys=ys,profts=profts)
    # Get the current date and time
    now = datetime.datetime.now()

    # Create a datetime object representing the current date and time

    # Display a message indicating what is being printed
    print("Current date and time : ")

    # Print the current date and time in a specific format
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    
    
def fasesep_prote1n_Kscale(Q,K,h0,tau2off,sigmaon,x1,nx,profilinterval,profilyn,deltax,dt,t_max,simnum):
    starttime=time.time()

    Q=float(Q)
    
    K=float(K)
    tau2on=tau2off/3
    taur= tau2on/tau2off
    
    dt0=dt
    x0=0
    lr=1/3
    nt=int(t_max/(dt0*profilinterval))
    
    tau2on=float(tau2on)
    tau2off=float(tau2off)
    sigmaon=float(sigmaon)
    
    h0=float(h0)
    
    #% Lag mesh
    dt=Constant(dt)
    deltax=(x1-x0)/nx

    mesh = RectangleMesh( Point(x0,x0), Point(x1, x1), nx, nx, diagonal='right')
    #mesh = generate_mesh(domain)
    #plot(mesh)
    class Periodic_sides(SubDomain):
        def inside(self, x, on_boundary):
            return bool((near(x[0], 0) or near(x[1], 0)) and (not ((near(x[0],0) and near(x[1],x1)) or (near(x[0],x1) and near(x[1],0)))) and on_boundary)

        def map(self, x, y):
            if near(x[0],x1) and near(x[1],x1):
                y[0] = x[0] - x1
                y[1] = x[1] - x1
            elif near(x[0],x1):
                y[0] = x[0] - x1
                y[1] = x[1]
            elif near(x[1],x1):
                y[0] = x[0]
                y[1] = x[1] - x1
            else:
                y[0]=-1000
                y[1]=-1000  
        
        
    #% Define mixed function space
    EL = FiniteElement('P',triangle,1)
    element = MixedElement([EL,EL,EL])
    VWS = FunctionSpace(mesh,element, constrained_domain=Periodic_sides())
    # Define initial condition 
    hpm_0 = Expression(('h0','0','0'), degree=2,x1=x1,h0=h0)

    # Define trial and test functions within function space
    v, w,s = TestFunctions(VWS)
    hpm = Function(VWS)
    h,p,m, = split(hpm)
    # Initialize solution in our function space at t=0
    hpm_n=project(hpm_0, VWS)
    h_n, p_n,m_n = split(hpm_n)
    # Make eta a function
    elmts = VectorElement('Lagrange', mesh.ufl_cell(), 1)
    etaV=FunctionSpace(mesh,elmts)
    eta=Function(etaV)
    
    ## Formulate the variational problem
    # Write out weak form of the equations with all terms on one side
    #
    #gammel med feilfaktorF=h_n*v*dx -h*v*dx-dt*Q*h**(3/2)*dot(grad(v),eta)*dx-dt*h**3*(1/(12))*dot(grad(p),grad(v))*dx +p*w*dx +B*dot(grad(w),grad(m))*dx -1*w*(exp(-((lr-h)/(sigmaon*lr))**2)/(exp(-((lr-h)/(sigmaon*lr))**2)+taur))*(h-lr)*dx - 1*w*lr*(exp(-((1-h)/(sigmaon*1))**2)/(exp(-((1-h)/(sigmaon*1))**2)+taur))*(h-1)*dx  +m*s*dx +dot(grad(s),grad(h))*dx   #+z1*c1_n*dx -z1*c1*dx +dt*z1*dot(grad(p),grad(c1))*(h*l1/mu)*dx +dt*(c10-c1)*z1*((1/tau1on)*exp(((l1-h)/(sigmaon*l1))**2))*dx -dt*c1*z1*((1/tau1off)*exp(((l1-h)/(sigmaoff*l1))**2))*dx -dt*D1*dot(grad(z1),grad(c1))*dx   +z2*c2_n*dx -z2*c2*dx +dt*z2*dot(grad(p),grad(c2))*(h*l2/mu)*dx +dt*(c20-c2)*z2*((1/tau2on)*exp(((l2-h)/(sigmaon*l2))**2))*dx -dt*c2*z2*((1/tau2off)*exp(((l2-h)/(sigmaoff*l2))**2))*dx -dt*D2*dot(grad(z2),grad(c2))*dx     
    F=h_n*v*dx -h*v*dx-dt*Q*h**(3/2)*dot(grad(v),eta)*dx-dt*h**3*dot(grad(p),grad(v))*dx +p*w*dx +dot(grad(w),grad(m))*dx - K*w*1*(exp(-((1-h)/(sigmaon*1))**2)/(exp(-((1-h)/(sigmaon*1))**2)+taur))*(h-1)*dx  +m*s*dx +dot(grad(s),grad(h))*dx   #+z1*c1_n*dx -z1*c1*dx +dt*z1*dot(grad(p),grad(c1))*(h*l1/mu)*dx +dt*(c10-c1)*z1*((1/tau1on)*exp(((l1-h)/(sigmaon*l1))**2))*dx -dt*c1*z1*((1/tau1off)*exp(((l1-h)/(sigmaoff*l1))**2))*dx -dt*D1*dot(grad(z1),grad(c1))*dx   +z2*c2_n*dx -z2*c2*dx +dt*z2*dot(grad(p),grad(c2))*(h*l2/mu)*dx +dt*(c20-c2)*z2*((1/tau2on)*exp(((l2-h)/(sigmaon*l2))**2))*dx -dt*c2*z2*((1/tau2off)*exp(((l2-h)/(sigmaoff*l2))**2))*dx -dt*D2*dot(grad(z2),grad(c2))*dx     

    J=derivative(F,hpm)

    problem=NonlinearVariationalProblem(F,hpm,J=J)
    solver=NonlinearVariationalSolver(problem)
    
    #% lag filer der data kan lagres
    t_series=open('t.txt','w')   # create and open text files in which to write down parameter values
    t_profiles=open('t_profil.txt','w')
    h_series=open('h.txt','w')
    xdmffile = XDMFFile('solution.xdmf') #create xdmf file for the solution
    xdmffile.parameters["flush_output"] = True
    profiles=np.zeros((nx+1,nx+1,nt+1))
    ts=np.zeros(nt)
    xs=np.zeros(nx+1)
    ys=np.zeros(nx+1)
    profts=np.zeros((nt+1))
    x=mesh.coordinates()
    xs=x[0:nx+1,0]
    ys=x[0:nx+1,0]
    nxd=nx
    xd=np.linspace(x0,x1,num=nxd)
    min=1000
    max=1000
    xmin=0
    for i1 in range(nxd-1):
        for i2 in range(nxd-1):
            val=hpm_n(xd[i1],xd[i2])[0]
            if val < min:
                min=val
                idmx=i1
                idmy=i2
    xmin=xd[idmx]
    ymin=xd[idmy]
    h_series.write(str(round(min,6)))       # write value
    h_series.write('\n') 


    def stochastic(Number,deltax,dt):
        mesh = UnitIntervalMesh(Number)        # lag mesh
        elms = FiniteElement('Lagrange', mesh.ufl_cell(), 1)  #
        U = FunctionSpace(mesh, elms)     # funksjonsrom,
        u = TrialFunction(U)
        v = TestFunction(U)
        eta = Function(U)
        stdevinv = deltax*np.sqrt(dt)
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
        
    #% registrer metadata
    simpars=open('simparametere.txt','w')
    simpars.write('nx = '+str(nx))
    simpars.write('\n')
    simpars.write('x0 = '+str(x0))
    simpars.write('\n')
    simpars.write('x1 = '+str(x1))
    simpars.write('\n')
    simpars.write('dt = '+str(dt0))
    simpars.write('\n')
    simpars.write('Q = '+str(Q))
    simpars.write('\n')
    simpars.write('K = '+str(K))
    simpars.write('\n')
    simpars.write('h_0 = '+str(h0))
    simpars.write('\n')

    #% Initialisér løsning
    F0=h*v*dx-h_n*v*dx +p*w*dx-p_n*w*dx   +m*s*dx -m_n*s*dx 
    solve(F0==0,hpm)
    
    #% Loop i tid
    t=0
    profts[0]=t
    _h0,_p0, _m0= hpm.split()
    xdmffile.write(_h0,t)
    
    
    t_series.write(str(round(t,6)))
    t_series.write('\n')
    t_profiles.write(str(round(t,6)))
    t_profiles.write('\n')

    for ix in range(nx+1):
        for iy in range(nx+1):
            profiles[ix,iy,0]=h0


    itno_old =0
    i=0
    iprof=0
    imax=-1000
    vals = np.zeros(2)
    klokk0=0
    klokk1=0
    klokk2=0
    while t<t_max:
        i+=1
        print('\n   \n')
        print( 'Q = '+str(Q)+', h0 = '+str(h0)+', K = '+str(K)+' -- Sim nummer '+str(simnum))
        klokk2=klokk1
        klokk1=klokk0
        klokk0=time.time()
        klokkdelt=np.mean([klokk0-klokk1,klokk1-klokk2])
        irem=t_max/dt0-i
        trem=klokkdelt*irem
        print('Estimated time remaining = '+str(datetime.timedelta(seconds=klokkdelt*irem)))
        # update current timestep
        t+=float(dt)                       # Time for which we will be computing solution
        print('Iteration number: '+str(i))#+'  /'+str(Nt))
        print('t = '+str(round(float(t),7)))#+'  /'+str(T))
        stochastic(len(eta.vector()),deltax,dt0)    # oppdater eta i tekstfil
        
        etaraw=np.loadtxt('u.txt') # last inn eta
        etaraw=etaraw[:,1]          # ta bare andre kolonne

        for ik in range(len(etaraw)-1):
            #print(ik)
            eta.vector()[ik] = etaraw[ik]         # sett vektorverdiene inn i funksjonen eta                         *********************
            if (np.sqrt(eta.vector()[ik]**2-etaraw[ik]**2)) != 0:
                print('Fail to assign new values to eta')


        # solve the variational problem to get new u at current timestep
        itno,conv =solver.solve()
        
        _h,_p, _m= hpm.split()
    
        hpm_n.assign(hpm)
        t_series.write(str(t))
        t_series.write('\n')
        

        if profilyn==1 and i/profilinterval==int(i/profilinterval):
            iprof+=1
            t_profiles.write(str(round(t,6)))
            t_profiles.write('\n')
            profts[iprof]=t
            for ix in range(nx+1):
                for iy in range(nx+1):
                    profiles[ix,iy,iprof]=_h(xs[ix],ys[iy])
            xdmffile.write(_h,t)
            
        min=1000
        for i1 in range(nxd-1):
            for i2 in range(nxd-1):
                val=hpm_n(xd[i1],xd[i2])[0]
                if val < min:
                    min=val
                    idmx=i1
                    idmy=i2
        h_series.write(str(round(min,6)))       # write value
        h_series.write('\n') 
        print('minimum height = '+str(min))
    #%
    h_series.close()
    t_series.close()
    t_profiles.close()

    endtime=time.time()
    print('Elapsed time = '+str(datetime.timedelta(seconds=endtime-starttime)))
    simpars.write('Elapsed real time = '+str(datetime.timedelta(seconds=endtime-starttime)))
    simpars.write('\n')
    simpars.write('# of iterations = '+str(i))
    simpars.write('\n')
    simpars.write('Simulation end time = '+str(t))
    simpars.close()

    #%
    np.savez('profiles.npz',profiles=profiles,xs=xs,ys=ys,profts=profts)
    # Get the current date and time
    now = datetime.datetime.now()

    # Create a datetime object representing the current date and time

    # Display a message indicating what is being printed
    print("Current date and time : ")

    # Print the current date and time in a specific format
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
def kontur(xs,ys,profts,profiles):
    figind=-1
    plt.figure()
    plt.contourf(xs,ys,profiles[:,:,figind],15)
    plt.title('t='+str(profts[figind])) 
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.savefig('sluttkontur.eps',bbox_inches='tight')    
    plt.close()

def Ndomains(xs,ys,profts,thresh,t0,t1,Nrunder):
    cwd = os.getcwd()
    
    exponent=0
    Ndoms=np.zeros(len(profts))
    xs=xs[:-1]
    ys=ys[:-1]
    n=xs/1
    N=len(n)
    print(Nrunder)
    for n in range(Nrunder):
        mappe='sim_'+str(n) 
        path = os.path.join(cwd,mappe)
        os.chdir(path)
        file_path = os.path.join(path, 'profiles.npz')
        if os.path.exists(file_path):
            profilfil=np.load('profiles.npz')
            profiles=profilfil['profiles']
            for it in range(len(profts)):
                h=profiles[:-1,:-1,it]
                h_bw = (h > thresh)
                Ndoms[it]+=count_and_plot_periodic_blobs(h_bw)
        #print(Ndoms[it])
    Ndoms=Ndoms/Nrunder
    os.chdir(cwd)
    Nt=len(profts)
    dt=profts[1]-profts[0]
    it0=round(t0/dt)
    it1=round(t1/dt)
    fit=np.polyfit(np.log(profts[it0:it1]), np.log(Ndoms[it0:it1]), 1)
    exponent=fit[0]
    #print('exponent='+str(exponent))
    fig1=plt.figure()
    ax=plt.axes()
    ax.plot(profts,Ndoms,'rx',label='fft')
    #ax.plot(ts,kchar1s,'bx',label='fft')
    ax.plot(profts[it0:it1],(exp(fit[1])/(1))*profts[it0:it1]**exponent,'--b',linewidth=1.5,label='$\propto t^{'+str(round(exponent,2))+'}$')
    ax.set_xlabel('t')
    ax.set_ylabel('N(t)')
    #ax.set_title('Q = '+str(Q))
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(loc='upper right')
    plt.savefig('Ndoms_logfit_thresh = '+str(thresh)+'.eps',bbox_inches='tight')
    plt.close()
    return Ndoms,exponent


def count_and_plot_periodic_blobs(binary_array):
    from scipy.ndimage import label
    from matplotlib.patches import Rectangle
    # Label the blobs in the binary array
    labeled_array, num_features = label(binary_array)

    # Arrays to track merged blobs
    blob_merges = {}
    for i in range(1, num_features + 1):
        blob_merges[i] = i

    # Merge blobs touching the top edge with blobs touching the bottom edge
    top_edge_labels = labeled_array[0, :]
    bottom_edge_labels = labeled_array[-1, :]
    for top, bottom in zip(top_edge_labels[top_edge_labels > 0], bottom_edge_labels[bottom_edge_labels > 0]):
        if top != bottom:
            blob_merges[bottom] = blob_merges[top]

    # Merge blobs touching the left edge with blobs touching the right edge
    left_edge_labels = labeled_array[:, 0]
    right_edge_labels = labeled_array[:, -1]
    for left, right in zip(left_edge_labels[left_edge_labels > 0], right_edge_labels[right_edge_labels > 0]):
        if left != right:
            blob_merges[right] = blob_merges[left]

    # Fix merging mapping to ensure every blob is merged to the lowest number label
    for k in blob_merges:
        while blob_merges[k] != blob_merges[blob_merges[k]]:
            blob_merges[k] = blob_merges[blob_merges[k]]

    # Renumber blobs and make a new array for visualization
    unique_blobs = set(blob_merges.values())
    new_labels = {old: new for new, old in enumerate(unique_blobs, start=1)}
    visual_array = np.copy(labeled_array)
    visual_labels = np.zeros_like(labeled_array)
    for old, new in new_labels.items():
        visual_labels[labeled_array == old] = new
    
    # Plot the original image with numbered labels
# =============================================================================
#     plt.figure(figsize=(10, 10))
#     plt.imshow(binary_array, cmap='gray', interpolation='none')
#     for label_value in range(1, num_features + 1):
#         positions = np.where(visual_labels == label_value)
#         if positions[0].size > 0:
#             y = positions[0][0]
#             x = positions[1][0]
#             plt.text(x, y, str(label_value), color="blue", fontsize=15, ha='center', va='center')
# 
#     plt.title("Original Image with Numbered Blobs")
#     plt.show()
# =============================================================================
    return len(unique_blobs)
def flekkemåling(xs,ys,profts,profiles,deltax,Q,hcut):
    figind=0
    sizefactors=np.zeros(len(profiles[0,0,:]))
    areas=np.zeros(len(profiles[0,0,:]))
    edges=np.zeros(len(profiles[0,0,:]))
    for it in range(len(profiles[0,0,:])):
        xgrad=np.gradient(profiles[:,:,figind])[0]
        ygrad=np.gradient(profiles[:,:,figind])[1]   
        gradsize=np.sqrt(xgrad**2+ygrad**2)
        edgepoints=np.zeros((len(gradsize[:,0]),len(gradsize[0,:])))
        domainpoints=np.zeros((len(gradsize[:,0]),len(gradsize[0,:])))  
        # =============================================================================
        # n_alene=42
        # cutoff=0.97
        # while n_alene>0:
        #     cutoff=cutoff-0.01
        #     for ix in range(len(gradsize[:,0])):
        #         for iy in range(len(gradsize[0,:])):
        #             if gradsize[ix,iy]>cutoff*np.max(gradsize):
        #                 edgepoints[ix,iy]=1
        #     n_alene=0
        #     for ix in range(len(gradsize[1:-1,0])):
        #         for iy in range(len(gradsize[0,1:-1])):
        #             nabosum=edgepoints[ix+1,iy]+edgepoints[ix-1,iy]+edgepoints[ix,iy+1]+edgepoints[ix,iy-1]
        #             if edgepoints[ix,iy]==1 and nabosum<2:
        #                 n_alene+=1
        #                 nabosum=0
        # for ix in range(len(gradsize[:,0])):
        #     for iy in range(len(gradsize[0,:])):
        #         nabosum=edgepoints[ix+1,iy]+edgepoints[ix-1,iy]+edgepoints[ix,iy+1]+edgepoints[ix,iy-1]
        #         if edgepoints[ix,iy]==1 and nabosum>2:
        # 
        # =============================================================================
        for ix in range(len(gradsize[:,0])):
            for iy in range(len(gradsize[0,:])):
                if profiles[ix,iy,figind] >hcut:
                    domainpoints[ix,iy]=1
                    
        for ix in range(len(gradsize[:-1,0])):
            for iy in range(len(gradsize[0,:])):
                if domainpoints[ix,iy]==1:
                    if domainpoints[ix+1,iy] != 1:
                        edgepoints[ix+1,iy]=1                
        for ix in range(len(gradsize[1:,0])):
            for iy in range(len(gradsize[0,:])):
                if domainpoints[ix,iy]==1:
                    if domainpoints[ix-1,iy] != 1:
                        edgepoints[ix-1,iy]=1
        for ix in range(len(gradsize[:,0])):
            for iy in range(len(gradsize[0,:-1])):
                if domainpoints[ix,iy]==1:
                    if domainpoints[ix,iy+1] != 1:
                        edgepoints[ix,iy+1]=1
        for ix in range(len(gradsize[:,0])):
            for iy in range(len(gradsize[0,1:])):
                if domainpoints[ix,iy]==1:
                    if domainpoints[ix,iy-1] != 1:
                        edgepoints[ix,iy-1]=1         
        if np.sum(domainpoints)>0:    
            sizefactor=deltax*np.sum(domainpoints)/np.sum(edgepoints)
        else:
            sizefactor=np.nan
        sizefactors[figind]=sizefactor
        edges[figind]=np.sum(edgepoints)
        areas[figind]=np.sum(domainpoints)
        figind+=1
    #print('Size factor = '+str(sizefactor))
# =============================================================================
#
#    if sizefactors[-1]>0:
#         plt.figure()
#         plt.plot(profts,sizefactors)
#         plt.xlabel('t')
#         plt.ylabel('a')
#         plt.title('Q = '+str(Q))
#         plt.savefig('sizefactorvt_linear.eps',bbox_inches='tight')
#         plt.close()
#     
#         fig1=plt.figure()
#         ax=plt.axes()
#         ax.plot(profts,sizefactors)
#         ax.set_xlabel('t')
#         ax.set_ylabel('a')
#         ax.set_title('Q = '+str(Q))
#         ax.set_yscale('log')
#     
#         ax.set_xscale('log')
#         plt.savefig('sizefactorvt_log.eps',bbox_inches='tight')
#         plt.close()
#     
# =============================================================================
    
    fig1=plt.figure()
    ax=plt.axes()
    ax.set_xscale('linear')
    ax.imshow(domainpoints,cmap='gray')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Q = '+str(Q))

    plt.savefig('domainpoints_hcut='+str(hcut)+'.eps',bbox_inches='tight')
    plt.close()


    fig1=plt.figure()
    ax=plt.axes()
    ax.set_xscale('linear')
    ax.imshow(edgepoints,cmap='gray')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Q = '+str(Q))

    plt.savefig('edgepoints_hcut='+str(hcut)+'.eps',bbox_inches='tight')
    plt.close()
    
    return sizefactors, areas, edges

def flekkemåling_gjsnitt(xs,ys,profts,profiles,deltax,hcut,Nrunder,t0,t1):
    print('begynner gjennomsnittsflekkemåling')
    cwd = os.getcwd()
    sizefactors=np.zeros(len(profiles[0,0,:]))
    exponent=0
    
    All_edges=np.zeros((len(sizefactors),Nrunder))
    All_areas=np.zeros((len(sizefactors),Nrunder))
    for n in range(Nrunder):
        mappe='sim_'+str(n) 
        #print( 'Simulering # '+str(n) +' - Laster inn til å kalkulere effektiv radius')
        path = os.path.join(cwd,mappe)
        os.chdir(path)
        file_path = os.path.join(path, 'sizefactors.npz')
        if os.path.exists(file_path):
            datafil=np.load(file_path)
            edges=datafil['edges']
            areas=datafil['areas']
            #print(areas)
            All_edges[:,n]=edges
            All_areas[:,n]=areas
            
        else:
            print('Fant ikke profiler - bruker forrige dobbelt')
    os.chdir(cwd)
    
    for it in range(len(profiles[0,0,:])):
        sizefactors[it]=deltax*np.sum(All_areas[it,:])/(np.sum(All_edges[it,:]))
    
    dt=profts[1]-profts[0]
    it0=round(t0/dt)
    it1=round(t1/dt)
    fit=np.polyfit(np.log(profts[it0:it1]), np.log(sizefactors[it0:it1]), 1)
    exponent=fit[0]
    #print('exponent='+str(exponent))
    fig1=plt.figure()
    ax=plt.axes()
    ax.plot(profts,sizefactors,'rx',label='Effective radius')
    #ax.plot(ts,kchar1s,'bx',label='fft')
    ax.plot(profts[it0:it1],(exp(fit[1])/(1))*profts[it0:it1]**exponent,'--b',linewidth=1.5,label='$\propto t^{'+f'{exponent:.3f}'+'}$')

    ax.set_xlabel('t')
    ax.set_ylabel('a')
    #ax.set_title('Q = '+str(Q))
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(loc='lower right')
    plt.savefig('averaged_effective_radii_t='+str(round(t0))+'-'+str(round(t1))+'_hcut='+str(hcut)+'.eps',bbox_inches='tight')
    plt.close()
    np.savez('sizefactors_gjsnitt.npz',sizefactors=sizefactors,All_areas=All_areas,All_edges=All_edges)
    
    
    
    print('gjennomsnittsflekkemåling ferdig')
    return sizefactors,exponent 
        
    

def kcalc_indS(xs,ys,profts,profiles,deltax,Q,hcut):
    figind=0
    sizefactors=np.zeros(len(profiles[0,0,:]))
    xs=xs[:-1]
    ys=ys[:-1]
    
    n=xs/1
    N=len(n)
    k=2*np.pi*n/N
    #kchar1s=np.zeros(len(profts))
    kchars=np.zeros(len(profts))
    for it in range(len(profiles[0,0,:])):
        #print('t='+str(profts[it]))
        
        h=profiles[:-1,:-1,it]
        psi=h-np.mean(h)
        
        ####################### AUTOCORRELATION + SHELL AVERAGE + FFT ###############
        #% Αutocorrelation in real space gives G
        #         
        #         G=np.zeros((len(k),len(k)))
        #         for ix in range(len(xs)): 
            #             for iy in range(len(ys)):
                #                 #print('ix = '+str(ix)+', iy = '+str(iy))
                #                 sum =0
                #                 counter=0
                #                 for jx in range(len(xs)):
                    #                     for jy in range(len(ys)):
                        #                         if ((jx+ix)<len(xs)) and ((jy+iy)<len(ys)):
                            #                             counter+=1
                            #                             partsum=psi[jx,jy]*psi[jx+ix,jy+iy]
                            #                             sum+=partsum
                            #                 G[ix,iy]=sum/counter
                            #                 
                            #     #% fft of G gives S
                            #         S=np.fft.fft2(G)
                            #         
                            #         #% Shell averaging
                            #         Sav=np.zeros(len(k),dtype = 'complex_')
                            #         kav=np.zeros(len(k))
                            #         
                            #         delta=2*pi/len(xs)
                            #         
                            #         for ik in range(len(k)):
                                #             Ssum=0
                                #             counter=0
                                #             ksum=0
                                #             for ikx in range(len(k)):
                                    #                  for iky in range(len(k)):
                                        #                      if k[ik]-delta/2 < np.sqrt(k[ikx]**2+k[iky]**2)< k[ik]+delta/2:
                                            #                          counter+=1 
                                            #                          Ssum+= S[ikx,iky]
                                            #                          ksum+=np.sqrt(k[ikx]**2+k[iky]**2)
                                            #             Sav[ik]=Ssum/counter
                                            #             kav[ik]=ksum/counter
                                            # 
                                            #         #% Compute final result -- characteristic wavenumber
                                            #         
                                            #         tel=0 
                                            #         nev=0 
                                            #         for ikx in range(len(k)):
                                                #              for iky in range(len(k)):
                                                    #                  if (k[ikx]**2+k[iky]**2)>0:
                                                        #                      tel+=np.abs(S[ikx,iky])/np.sqrt(k[ikx]**2+k[iky]**2)
                                                        #                      nev+=np.abs(S[ikx,iky])/(k[ikx]**2+k[iky]**2)
                                                        #         
                                                        #         if nev>0:
                                                            #             kchar1=tel/nev
                                                            #         else:
                                                                #             kchar1=np.nan
                                                                #             
                                                                # =============================================================================
                                                                #################################### FFT THEN MULTIPLY #########
                                                                #% FFT of psi - gives psik from article
                                                                
        
        # =============================================================================
        psiknp=np.fft.fft2(psi)
        #%
        
        Sfnp=np.zeros((len(k),len(k)),dtype = 'complex_')
        for ikx in range(len(k)):
            for iky in range(len(k)):
                Sfnp[ikx,iky]=psiknp[ikx,iky]*psiknp[-ikx,-iky]
    #% Compute final result -- characteristic wavenumber
        
        tel=0 
        nev=0 
        for ikx in range(len(k)):
             for iky in range(len(k)):
                 if (k[ikx]**2+k[iky]**2)>0:
                     tel+=np.abs(Sfnp[ikx,iky])/np.sqrt(k[ikx]**2+k[iky]**2)
                     nev+=np.abs(Sfnp[ikx,iky])/(k[ikx]**2+k[iky]**2)
        
        if nev>0:
            kchar2np=tel/nev
        else:
            kchar2np=np.nan
            
        #print('longway = '+str(kchar1))
        #print('shortway = '+str(kchar2np))
        kchars[it]=kchar2np
        #kchar1s[it]=kchar1
        
    #%
    return kchars
    
def kcalc_Sav(xs,ys,profts,profiles,Nrunder,t0,t1):
    cwd = os.getcwd()
    figind=0
    xs=xs[:-1]
    ys=ys[:-1]
    n=xs/1
    N=len(n)
    k=2*np.pi*n/N
    kchars=np.zeros(len(profts))
    All_profiles=np.zeros((len(profiles[:,0,0]),len(profiles[0,:,0]),len(profiles[0,0,:]),Nrunder))
    for n in range(Nrunder):
        mappe='sim_'+str(n) 
        print( 'Simulering # '+str(n) +' - Laster inn til å kalkulere k')
        path = os.path.join(cwd,mappe)
        os.chdir(path)
        file_path = os.path.join(path, 'profiles.npz')
        if os.path.exists(file_path):
            profilfil=np.load('profiles.npz')
            profiles=profilfil['profiles']
            All_profiles[:,:,:,n]=profiles
        else:
            print('Fant ikke profiler - bruker forrige dobbelt')
    
    for it in range(len(profts)):
        Sav=np.zeros((len(profiles[:-1,0,0]),len(profiles[0,:-1,0])),dtype = 'complex_')
        for n in range(Nrunder):
            h=All_profiles[:-1,:-1,it,n]
            psi=h-np.mean(h)
      
            psik=np.fft.fft2(psi)
    
            S=np.zeros((len(k),len(k)),dtype = 'complex_')
            for ikx in range(len(k)):
                for iky in range(len(k)):
                    S[ikx,iky]=psik[ikx,iky]*psik[-ikx,-iky]
                    Sav[ikx,iky]+=S[ikx,iky]/Nrunder
        
    #% Compute final result -- characteristic wavenumber
        
        tel=0 
        nev=0 
        for ikx in range(len(k)):
             for iky in range(len(k)):
                 if (k[ikx]**2+k[iky]**2)>0:
                     tel+=np.abs(Sav[ikx,iky])/np.sqrt(k[ikx]**2+k[iky]**2)
                     nev+=np.abs(Sav[ikx,iky])/(k[ikx]**2+k[iky]**2)
        
        if nev>0:
            kchar=tel/nev
        else:
            kchar=np.nan
            
        kchars[it]=kchar
    os.chdir(cwd)  
    exponent=np.nan
    Nt=len(profts)
    dt=profts[1]-profts[0]
    it0=round(t0/dt)
    it1=round(t1/dt)
# =============================================================================
#     fit=np.polyfit(np.log(profts[it0:it1]), np.log(kchars[it0:it1]), 1)
#     exponent=fit[0]
#     #print('exponent='+str(exponent))
#     fig1=plt.figure()
#     ax=plt.axes()
#     ax.plot(profts,kchars,'rx',label='fft')
#     #ax.plot(ts,kchar1s,'bx',label='fft')
#     ax.plot(profts[it0:it1],(exp(fit[1])/(1))*profts[it0:it1]**exponent,'--b',linewidth=1.5,label='$\propto t^{'+str(round(exponent,2))+'}$')
#     ax.set_xlabel('t')
#     ax.set_ylabel('<k>')
#     #ax.set_title('Q = '+str(Q))
#     ax.set_yscale('log')
#     ax.set_xscale('log')
#     ax.legend(loc='upper right')
#     plt.savefig('kvt_Sav_logfit.eps',bbox_inches='tight')
#     plt.close()
#         
# =============================================================================
    #lamd1s=2*np.pi/kchar1s
    lamd2s=2*np.pi/kchars
    fit=np.polyfit(np.log(profts[it0:it1]), np.log(lamd2s[it0:it1]), 1)
    exponent=fit[0]
    fig1=plt.figure()
    ax=plt.axes()
    ax.plot(profts,lamd2s,'rx',label='fft')
    #ax.plot(ts,lamd1s,'bx',label='fft')
    ax.plot(profts[it0:it1],(exp(fit[1])/(1))*profts[it0:it1]**exponent,'--b',linewidth=1.5,label='$\propto t^{'+f'{exponent:.3f}'+'}$')
    ax.set_xlabel('t')
    ax.set_ylabel('a')
    #ax.set_title('Q = '+str(Q))
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(loc='lower right')
    plt.savefig('Vira_sizefact_old_'+str(round(t0))+'-'+str(round(t1))+'.eps',bbox_inches='tight')
    plt.close()
    
    return kchars,exponent


def kcalc_JBL(xs,ys,profts,profiles,Nrunder,t0,t1):
    cwd = os.getcwd()
    figind=0
    xs=xs[:-1]
    ys=ys[:-1]
    
    L=xs[-1]
    Nx=len(xs)
    kchars=np.zeros(len(profts))
    All_profiles=np.zeros((len(profiles[:,0,0]),len(profiles[0,:,0]),len(profiles[0,0,:]),Nrunder))
    freq = np.arange(0,Nx,1)*2*np.pi/L
    freq2d = np.meshgrid(freq,freq)
    freq2d_norm = np.sqrt(freq2d[0]**2+freq2d[1]**2)
    freq2d_norm[0,0] = 1
    for n in range(Nrunder):
        mappe='sim_'+str(n) 
        print( 'Simulering # '+str(n) +' - Laster inn til å kalkulere k')
        path = os.path.join(cwd,mappe)
        os.chdir(path)
        file_path = os.path.join(path, 'profiles.npz')
        if os.path.exists(file_path):
            profilfil=np.load('profiles.npz')
            profiles=profilfil['profiles']
            profts=profilfil['profts']
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
            All_profiles[:,:,:,n]=profiles
        else:
            print('Fant ikke profiler - bruker forrige dobbelt')
    for it in range(len(profts)):
        Sav=np.zeros((len(profiles[:-1,0,0]),len(profiles[0,:-1,0])),dtype = 'complex_')
        for n in range(Nrunder):
            h=All_profiles[:-1,:-1,it,n]
            psi=h-np.mean(h)
            # fft and calculate S
            S = np.abs(np.fft.fft2(psi)/Nx)**2
            S[0,0] = 0
            Sav+=S/Nrunder
            #np.savez('Sav_orig.npz',Sav=Sav)
            #exit()
            
        kchars[it]=np.sum(Sav/freq2d_norm)/np.sum(Sav/freq2d_norm**2)
    exponent=np.nan
    os.chdir(cwd) 
    Nt=len(profts)
    dt=profts[1]-profts[0]
    it0=round(t0/dt)
    it1=round(t1/dt)
    
    fit=np.polyfit(np.log(profts[it0:it1]), np.log(kchars[it0:it1]), 1)
    exponent=fit[0]
    #print('exponent='+str(exponent))
    fig1=plt.figure()
    ax=plt.axes()
    ax.plot(profts,kchars,'rx',label='fft')
    #ax.plot(ts,kchar1s,'bx',label='fft')
    ax.plot(profts[it0:it1],(exp(fit[1])/(1))*profts[it0:it1]**exponent,'--b',linewidth=1.5,label='$\propto t^{'+str(round(exponent,2))+'}$')
    ax.set_xlabel('t')
    ax.set_ylabel('<k>')
    #ax.set_title('Q = '+str(Q))
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(loc='upper right')
    plt.savefig('kvt_JBL_logfit.eps',bbox_inches='tight')
    plt.close()
        
    #lamd1s=2*np.pi/kchar1s
    lamd2s=2*np.pi/kchars
    fit=np.polyfit(np.log(profts[it0:it1]), np.log(lamd2s[it0:it1]), 1)
    exponent=fit[0]
    fig1=plt.figure()
    ax=plt.axes()
    ax.plot(profts,lamd2s,'rx',label='fft')
    #ax.plot(ts,lamd1s,'bx',label='fft')
    ax.plot(profts[it0:it1],(exp(fit[1])/(1))*profts[it0:it1]**exponent,'--b',linewidth=1.5,label='$\propto t^{'+f'{exponent:.3f}'+'}$')
    ax.set_xlabel('t')
    ax.set_ylabel('a')
    #ax.set_title('Q = '+str(Q))
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(loc='lower right')
    plt.savefig('SO_growth_t='+str(round(t0))+'-'+str(round(t1))+'.eps',bbox_inches='tight')
    plt.close()
    np.savez('Sizefactorsvt_SO.npz',t=profts,L=lamd2s)
    return kchars,exponent    


def nequals1_(xs,ys,profts,profiles,Nrunder,t0,t1):
    cwd = os.getcwd()
    figind=0
    xs=xs[:-1]
    ys=ys[:-1]
    
    L=xs[-1]
    Nx=len(xs)
    kchars=np.zeros(len(profts))
    All_profiles=np.zeros((len(profiles[:,0,0]),len(profiles[0,:,0]),len(profiles[0,0,:]),Nrunder))
    freq = np.arange(0,Nx,1)*2*np.pi/L
    freq2d = np.meshgrid(freq,freq)
    freq2d_norm = np.sqrt(freq2d[0]**2+freq2d[1]**2)
    freq2d_norm[0,0] = 1
    for n in range(Nrunder):
        mappe='sim_'+str(n) 
        print( 'Simulering # '+str(n) +' - Laster inn til å kalkulere k')
        path = os.path.join(cwd,mappe)
        os.chdir(path)
        file_path = os.path.join(path, 'profiles.npz')
        if os.path.exists(file_path):
            profilfil=np.load('profiles.npz')
            profiles=profilfil['profiles']
            profts=profilfil['profts']
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
            All_profiles[:,:,:,n]=profiles
        else:
            print('Fant ikke profiler - bruker forrige dobbelt')
    for it in range(len(profts)):
        Sav=np.zeros((len(profiles[:-1,0,0]),len(profiles[0,:-1,0])),dtype = 'complex_')
        for n in range(Nrunder):
            h=All_profiles[:-1,:-1,it,n]
            psi=h-np.mean(h)
            # fft and calculate S
            S = np.abs(np.fft.fft2(psi)/Nx)**2
            S[0,0] = 0
            Sav+=S/Nrunder
            
        kchars[it]=np.sum(Sav*freq2d_norm)/np.sum(Sav)
    exponent=np.nan
    os.chdir(cwd) 
    Nt=len(profts)
    dt=profts[1]-profts[0]
    it0=round(t0/dt)
    it1=round(t1/dt)
    
# =============================================================================
#     fit=np.polyfit(np.log(profts[it0:it1]), np.log(kchars[it0:it1]), 1)
#     exponent=fit[0]
#     #print('exponent='+str(exponent))
#     fig1=plt.figure()
#     ax=plt.axes()
#     ax.plot(profts,kchars,'rx',label='fft')
#     #ax.plot(ts,kchar1s,'bx',label='fft')
#     ax.plot(profts[it0:it1],(exp(fit[1])/(1))*profts[it0:it1]**exponent,'--b',linewidth=1.5,label='$\propto t^{'+str(round(exponent,2))+'}$')
#     ax.set_xlabel('t')
#     ax.set_ylabel('<k>')
#     #ax.set_title('Q = '+str(Q))
#     ax.set_yscale('log')
#     ax.set_xscale('log')
#     ax.legend(loc='upper right')
#     plt.savefig('kvt_JBL_logfit.eps',bbox_inches='tight')
#     plt.close()
# =============================================================================
        
    #lamd1s=2*np.pi/kchar1s
    lamd2s=2*np.pi/kchars
    fit=np.polyfit(np.log(profts[it0:it1]), np.log(lamd2s[it0:it1]), 1)
    exponent=fit[0]
    fig1=plt.figure()
    ax=plt.axes()
    ax.plot(profts,lamd2s,'rx',label='fft')
    #ax.plot(ts,lamd1s,'bx',label='fft')
    ax.plot(profts[it0:it1],(exp(fit[1])/(1))*profts[it0:it1]**exponent,'--b',linewidth=1.5,label='$\propto t^{'+f'{exponent:.3f}'+'}$')
    ax.set_xlabel('t')
    ax.set_ylabel('a')
    #ax.set_title('Q = '+str(Q))
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim([lamd2s[1], lamd2s[-1]])
    ax.legend(loc='upper left')
    plt.savefig('growthlaw_nequals1_'+str(round(t0))+'-'+str(round(t1))+'.eps',bbox_inches='tight')
    plt.close()
    np.savez('Sizefactorsvt_nequals1.npz',t=profts,L=lamd2s)
    return kchars,exponent  

def nequals2_(xs,ys,profts,profiles,Nrunder,t0,t1):
    cwd = os.getcwd()
    figind=0
    xs=xs[:-1]
    ys=ys[:-1]
    
    L=xs[-1]
    Nx=len(xs)
    kchars=np.zeros(len(profts))
    All_profiles=np.zeros((len(profiles[:,0,0]),len(profiles[0,:,0]),len(profiles[0,0,:]),Nrunder))
    freq = np.arange(0,Nx,1)*2*np.pi/L
    freq2d = np.meshgrid(freq,freq)
    freq2d_norm = np.sqrt(freq2d[0]**2+freq2d[1]**2)
    freq2d_norm[0,0] = 1
    for n in range(Nrunder):
        mappe='sim_'+str(n) 
        print( 'Simulering # '+str(n) +' - Laster inn til å kalkulere k')
        path = os.path.join(cwd,mappe)
        os.chdir(path)
        file_path = os.path.join(path, 'profiles.npz')
        if os.path.exists(file_path):
            profilfil=np.load('profiles.npz')
            profiles=profilfil['profiles']
            profts=profilfil['profts']
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
            All_profiles[:,:,:,n]=profiles
        else:
            print('Fant ikke profiler - bruker forrige dobbelt')
    for it in range(len(profts)):
        Sav=np.zeros((len(profiles[:-1,0,0]),len(profiles[0,:-1,0])),dtype = 'complex_')
        for n in range(Nrunder):
            h=All_profiles[:-1,:-1,it,n]
            psi=h-np.mean(h)
            # fft and calculate S
            S = np.abs(np.fft.fft2(psi)/Nx)**2
            S[0,0] = 0
            Sav+=S/Nrunder
            
        kchars[it]=(np.sum(Sav*freq2d_norm**2)/np.sum(Sav))**(1/2)
    exponent=np.nan
    os.chdir(cwd) 
    Nt=len(profts)
    dt=profts[1]-profts[0]
    it0=round(t0/dt)
    it1=round(t1/dt)
    
    fit=np.polyfit(np.log(profts[it0:it1]), np.log(kchars[it0:it1]), 1)
    exponent=fit[0]
    #print('exponent='+str(exponent))
# =============================================================================
#     fig1=plt.figure()
#     ax=plt.axes()
#     ax.plot(profts,kchars,'rx',label='fft')
#     #ax.plot(ts,kchar1s,'bx',label='fft')
#     ax.plot(profts[it0:it1],(exp(fit[1])/(1))*profts[it0:it1]**exponent,'--b',linewidth=1.5,label='$\propto t^{'+str(round(exponent,2))+'}$')
#     ax.set_xlabel('t')
#     ax.set_ylabel('<k>')
#     #ax.set_title('Q = '+str(Q))
#     ax.set_yscale('log')
#     ax.set_xscale('log')
#     ax.legend(loc='upper right')
#     plt.savefig('kvt_JBL_logfit.eps',bbox_inches='tight')
#     plt.close()
# =============================================================================
        
    #lamd1s=2*np.pi/kchar1s
    lamd2s=2*np.pi/kchars
    fit=np.polyfit(np.log(profts[it0:it1]), np.log(lamd2s[it0:it1]), 1)
    exponent=fit[0]
    fig1=plt.figure()
    ax=plt.axes()
    ax.plot(profts,lamd2s,'rx',label='fft')
    #ax.plot(ts,lamd1s,'bx',label='fft')
    ax.plot(profts[it0:it1],(exp(fit[1])/(1))*profts[it0:it1]**exponent,'--b',linewidth=1.5,label='$\propto t^{'+f'{exponent:.3f}'+'}$')
    ax.set_xlabel('t')
    ax.set_ylabel('a')
    #ax.set_title('Q = '+str(Q))
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim([lamd2s[1], lamd2s[-1]])
    ax.legend(loc='upper left')
    plt.savefig('growthlaw_nequals2_'+str(round(t0))+'-'+str(round(t1))+'.eps',bbox_inches='tight')
    plt.close()
    np.savez('Sizefactorsvt_nequals2.npz',t=profts,L=lamd2s)
    return kchars,exponent  


def shellavg(xs,ys,profts,profiles,Nrunder,t0,t1):
    cwd = os.getcwd()
    figind=0
    xs=xs[:-1]
    ys=ys[:-1]
    
    L=xs[-1]
    Nx=len(xs)
    kchars=np.zeros(len(profts))
    kchars2=np.zeros(len(profts))
    kchars3=np.zeros(len(profts))
    All_profiles=np.zeros((len(profiles[:,0,0]),len(profiles[0,:,0]),len(profiles[0,0,:]),Nrunder))
    freq = np.arange(0,Nx,1)*2*np.pi/L
    #k=2*np.pi*n/N
    #print(freq)
    #print(k)
    freq2d = np.meshgrid(freq,freq)
    freq2d_norm = np.sqrt(freq2d[0]**2+freq2d[1]**2)
    freq2d_norm[0,0] = 1
    for n in range(Nrunder):
        mappe='sim_'+str(n) 
        print( 'Simulering # '+str(n) +' - Laster inn til å kalkulere k')
        path = os.path.join(cwd,mappe)
        os.chdir(path)
        file_path = os.path.join(path, 'profiles.npz')
        if os.path.exists(file_path):
            profilfil=np.load('profiles.npz')
            profiles=profilfil['profiles']
            profts=profilfil['profts']
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
            All_profiles[:,:,:,n]=profiles
        else:
            print('Fant ikke profiler - bruker forrige dobbelt')
    for it in range(len(profts)):
        Sav=np.zeros((len(profiles[:-1,0,0]),len(profiles[0,:-1,0])),dtype = 'complex_')
        for n in range(Nrunder):
            h=All_profiles[:-1,:-1,it,n]
            psi=h-np.mean(h)
            # fft and calculate S
            S = np.abs(np.fft.fft2(psi)/Nx)**2
            S[0,0] = 0
            Sav+=S/Nrunder
        #np.savez('Sav.npz',Sav=Sav)
        #exit()
        #  Shell averaging
        Sshav=np.zeros(len(freq),dtype = 'complex_')
        kav=np.zeros(len(freq))
        delta=2*pi/len(xs)
        for ik in range(len(freq)):
            Ssum=0
            counter=0
            ksum=0
            for ikx in range(len(freq)):
                for iky in range(len(freq)):
                    if freq[ik]-delta/2 < np.sqrt(freq[ikx]**2+freq[iky]**2)< freq[ik]+delta/2:
                        counter+=1 
                        Ssum+= Sav[ikx,iky]
                        ksum+=np.sqrt(freq[ikx]**2+freq[iky]**2)
                        Sshav[ik]=Ssum/counter
                        kav[ik]=ksum/counter
        #print(Sshav)
        if it >0:
            ikpeak=np.argmax(Sshav)
            Sshav=Sshav[0:5*ikpeak]
            freq=freq[0:5*ikpeak]
        print('start integration'+'t='+str(profts[it]))
        kchars[it]=np.sum(Sshav*freq)/np.sum(Sshav)
        kchars2[it]=(np.sum(Sshav*freq**2)/np.sum(Sshav))**(1/2)
        kchars3[it]=(np.sum(Sshav*freq**3)/np.sum(Sshav))**(1/3)
    exponent=np.nan
    os.chdir(cwd) 
    Nt=len(profts)
    dt=profts[1]-profts[0]
    it0=round(t0/dt)
    it1=round(t1/dt)
    
# =============================================================================
#     fit=np.polyfit(np.log(profts[it0:it1]), np.log(kchars[it0:it1]), 1)
#     exponent=fit[0]
#     #print('exponent='+str(exponent))
#     fig1=plt.figure()
#     ax=plt.axes()
#     ax.plot(profts,kchars,'rx',label='fft')
#     #ax.plot(ts,kchar1s,'bx',label='fft')
#     ax.plot(profts[it0:it1],(exp(fit[1])/(1))*profts[it0:it1]**exponent,'--b',linewidth=1.5,label='$\propto t^{'+str(round(exponent,2))+'}$')
#     ax.set_xlabel('t')
#     ax.set_ylabel('<k>')
#     #ax.set_title('Q = '+str(Q))
#     ax.set_yscale('log')
#     ax.set_xscale('log')
#     ax.legend(loc='upper right')
#     plt.savefig('kvt_JBL_logfit.eps',bbox_inches='tight')
#     plt.close()
# =============================================================================
        
    #lamd1s=2*np.pi/kchar1s
    Ln1=2*np.pi/kchars
    fit1=np.polyfit(np.log(profts[it0:it1]), np.log(Ln1[it0:it1]), 1)
    exp1=fit1[0]
# =============================================================================
#     fig1=plt.figure()
#     ax=plt.axes()
#     ax.plot(profts,Ln1,'rx',label='n=1')
#     #ax.plot(ts,lamd1s,'bx',label='fft')
#     ax.plot(profts[it0:it1],(exp(fit1[1])/(1))*profts[it0:it1]**exp1,'--b',linewidth=1.5,label='$\propto t^{'+f'{exponent:.3f}'+'}$')
#     ax.set_xlabel('t')
#     ax.set_ylabel('a')
#     #ax.set_title('Q = '+str(Q))
#     ax.set_yscale('log')
#     ax.set_xscale('log')
#     ax.set_ylim([Ln1[1], Ln1[-1]])
#     ax.legend(loc='upper left')
#     plt.savefig('growthlaw_n=1_'+str(round(t0))+'-'+str(round(t1))+'.eps',bbox_inches='tight')
#     plt.close()
# =============================================================================
    #np.savez('Sizefactorsvt_n=1.npz',t=profts,L=Ln1)
    
    
    Ln2=2*np.pi/kchars2
    fit2=np.polyfit(np.log(profts[it0:it1]), np.log(Ln2[it0:it1]), 1)
    exp2=fit2[0]
# =============================================================================
#     fig1=plt.figure()
#     ax=plt.axes()
#     ax.plot(profts,Ln2,'rx',label='n=2')
#     #ax.plot(ts,lamd1s,'bx',label='fft')
#     ax.plot(profts[it0:it1],(exp(fit2[1])/(1))*profts[it0:it1]**exp2,'--b',linewidth=1.5,label='$\propto t^{'+f'{exponent:.3f}'+'}$')
#     ax.set_xlabel('t')
#     ax.set_ylabel('a')
#     #ax.set_title('Q = '+str(Q))
#     ax.set_yscale('log')
#     ax.set_xscale('log')
#     ax.set_ylim([Ln2[1], Ln2[-1]])
#     ax.legend(loc='upper left')
#     plt.savefig('growthlaw_n=2_'+str(round(t0))+'-'+str(round(t1))+'.eps',bbox_inches='tight')
#     plt.close()
# =============================================================================
    #np.savez('Sizefactorsvt_n=2.npz',t=profts,L=Ln2)
    
    Ln3=2*np.pi/kchars3
    fit3=np.polyfit(np.log(profts[it0:it1]), np.log(Ln3[it0:it1]), 1)
    exp3=fit3[0]
# =============================================================================
#     fig1=plt.figure()
#     ax=plt.axes()
#     ax.plot(profts,Ln3,'rx',label='n=3')
#     #ax.plot(ts,lamd1s,'bx',label='fft')
#     ax.plot(profts[it0:it1],(exp(fit3[1])/(1))*profts[it0:it1]**exp3,'--b',linewidth=1.5,label='$\propto t^{'+f'{exponent:.3f}'+'}$')
#     ax.set_xlabel('t')
#     ax.set_ylabel('a')
#     #ax.set_title('Q = '+str(Q))
#     ax.set_yscale('log')
#     ax.set_xscale('log')
#     ax.set_ylim([Ln3[1], Ln3[-1]])
#     ax.legend(loc='upper left')
#     plt.savefig('growthlaw_n=3_'+str(round(t0))+'-'+str(round(t1))+'.eps',bbox_inches='tight')
#     plt.close()
# =============================================================================
    #np.savez('Sizefactorsvt_n=3.npz',t=profts,L=Ln2)
    np.savez('Shellavg_growth.npz',t=profts,Ln1=Ln1,Ln2=Ln2,Ln3=Ln3)
    
    
    fig1=plt.figure()
    ax=plt.axes()
    ax.plot(profts,Ln1,'bx',label='n=1', markersize=5)
    ax.plot(profts,Ln2,'ro',label='n=2', markersize=5)
    ax.plot(profts,Ln3,'g+',label='n=3', markersize=5)
    ax.plot(profts[it0:it1],(exp(fit1[1])/(1))*profts[it0:it1]**exp1,'-k',linewidth=1.5,label='$\propto t^{'+f'{exp1:.3f}'+'}$')
    ax.plot(profts[it0:it1],(exp(fit2[1])/(1))*profts[it0:it1]**exp2,'--k',linewidth=1.5,label='$\propto t^{'+f'{exp2:.3f}'+'}$')
    ax.plot(profts[it0:it1],(exp(fit3[1])/(1))*profts[it0:it1]**exp3,'-.k',linewidth=1.5,label='$\propto t^{'+f'{exp3:.3f}'+'}$')
    ax.set_xlabel('t')
    ax.set_ylabel('a')
    #ax.set_title('Q = '+str(Q))
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim([Ln1[1], Ln1[-1]])
    ax.legend(loc='upper left')
    plt.savefig('growthlaw_shellavg_compare_n'+str(round(t0))+'-'+str(round(t1))+'.eps',bbox_inches='tight')
    plt.close()
     


def int_energies_avg(xs,ys,profts,profiles,Nrunder,taur,sigmaon,deltax):
    scalsep=100
    cwd = os.getcwd()
    figind=0
    xs=xs[:]
    ys=ys[:]
    
    L=xs[-1]
    Nx=len(xs)
    Eints=np.zeros(len(profts))
    Ebinds=np.zeros(len(profts))
    Edisss=np.zeros(len(profts))
    All_profiles=np.zeros((len(profiles[:,0,0]),len(profiles[0,:,0]),len(profiles[0,0,:]),Nrunder))

    #All_profiles=np.zeros((len(profiles[:,0,0]),len(profiles[0,:,0]),len(profiles[0,0,:]),Nrunder))
    for n in range(Nrunder):
        mappe='sim_'+str(n) 
        print( 'Simulering # '+str(n) +' - Laster inn til å kalkulere energier')
        path = os.path.join(cwd,mappe)
        os.chdir(path)
        file_path = os.path.join(path, 'profiles.npz')
        if os.path.exists(file_path):
            profilfil=np.load('profiles.npz')
            profiles=profilfil['profiles']
            All_profiles[:,:,:,n]=profiles
        else:
            print('Fant ikke profiler - bruker forrige dobbelt')
    for it in range(len(profts)):
        for n in range(Nrunder):
            h=All_profiles[:,:,it,n]
            
            gradh=np.zeros((len(h[:,0]),len(h[0,:])))
            
            gradh=np.gradient(h, xs,ys)
            dhdx=gradh[0]
            dhdy=gradh[1]
            
            #d2hdx2=np.gradient(dhdx, xs,ys)
            Eint_el=(scalsep)*np.sqrt((scalsep**2)+dhdx**2+dhdy**2)
            #Eint_el=(dhdx**2+dhdy**2)/2
            Eint=np.sum(Eint_el)*deltax*deltax
            
            Eints[it]+=Eint/Nrunder
            
            #Ebind_el=-((sigmaon**2)/2)*np.log(np.exp(-((1-h)/(sigmaon))**2)+taur)
            Ebind_el=(1/2)*((1-h)**2)*np.exp(-((1-h)/(sigmaon))**2)/(np.exp(-((1-h)/(sigmaon))**2)+taur)
            Ebind=np.sum(Ebind_el)*deltax*deltax
            Ebinds[it]+=Ebind/Nrunder
    np.savez('Energies.npz',Ebinds=Ebinds,Eints=Eints)
    dEints=Eints-Eints[0]
    #dEints=Eints
    os.chdir(cwd)
    np.savez('Energies.npz',Ebinds=Ebinds,dEints=dEints)
    plt.figure()
    plt.plot(profts,dEints,label='$\Delta E_\sigma$')
    plt.plot(profts,Ebinds,label='$E_{prot}$')
    plt.xlabel('t')
    plt.legend()
    plt.ylim(np.min((np.min(Ebinds),np.min(dEints))),np.max((np.max(Ebinds),np.max(dEints))))
    plt.savefig('Mean energies vs time'+'.eps',bbox_inches='tight')
    plt.close()
    plt.figure()
    plt.semilogx(profts,dEints,label='$\Delta E_\sigma$')
    plt.semilogx(profts,Ebinds,label='$E_{prot}$')
    plt.xlabel('t')
    plt.legend()
    plt.ylim(np.min(Ebinds), np.max(Ebinds[1:]))
    plt.savefig('Mean energies vs time -- semilogx'+'.eps',bbox_inches='tight')
    plt.close()
def bend_energies_avg(xs,ys,profts,profiles,Nrunder,taur,sigmaon,deltax):
    cwd = os.getcwd()
    figind=0
    xs=xs[:]
    ys=ys[:]
    
    L=xs[-1]
    Nx=len(xs)
    Eints=np.zeros(len(profts))
    Ebinds=np.zeros(len(profts))
    Edisss=np.zeros(len(profts))
    All_profiles=np.zeros((len(profiles[:,0,0]),len(profiles[0,:,0]),len(profiles[0,0,:]),Nrunder))

    #All_profiles=np.zeros((len(profiles[:,0,0]),len(profiles[0,:,0]),len(profiles[0,0,:]),Nrunder))
    for n in range(Nrunder):
        mappe='sim_'+str(n) 
        print( 'Simulering # '+str(n) +' - Laster inn til å kalkulere energier')
        path = os.path.join(cwd,mappe)
        os.chdir(path)
        file_path = os.path.join(path, 'profiles.npz')
        if os.path.exists(file_path):
            profilfil=np.load('profiles.npz')
            profiles=profilfil['profiles']
            All_profiles[:,:,:,n]=profiles
        else:
            print('Fant ikke profiler - bruker forrige dobbelt')
    for it in range(len(profts)):
        for n in range(Nrunder):
            h=All_profiles[:,:,it,n]
            
            gradh=np.zeros((len(h[:,0]),len(h[0,:])))
            
            gradh=np.gradient(h, xs,ys)
            dhdx=gradh[0]
            dhdy=gradh[1]
            
            #d2hdx2=np.gradient(dhdx, xs,ys)
            Eint_el=np.sqrt(1+dhdx**2+dhdy**2)/2
            
            Eint=np.sum(Eint_el)*deltax*deltax
            
            Eints[it]+=Eint/Nrunder
            
            Ebind_el=-((sigmaon**2)/2)*np.log(np.exp(-((1-h)/(sigmaon))**2)+taur)
            Ebind=np.sum(Ebind_el)*deltax*deltax
            Ebinds[it]+=Ebind/Nrunder
    np.savez('Energies.npz',Ebinds=Ebinds,Eints=Eints)
    dEints=Eints-Eints[0]
    os.chdir(cwd)
    np.savez('Energies.npz',Ebinds=Ebinds,dEints=dEints)
    plt.figure()
    plt.plot(profts,dEints,label='$\Delta E_\sigma$')
    plt.plot(profts,Ebinds,label='$E_{prot}$')
    plt.xlabel('t')
    plt.legend()
    plt.ylim(np.min(Ebinds), 1.5*np.max(dEints))
    plt.savefig('Mean energies vs time'+'.eps',bbox_inches='tight')
    plt.close()
    plt.figure()
    plt.semilogx(profts,dEints,label='$\Delta E_\sigma$')
    plt.semilogx(profts,Ebinds,label='$E_{prot}$')
    plt.xlabel('t')
    plt.legend()
    plt.ylim(np.min(Ebinds), np.max(Ebinds[1:]))
    plt.savefig('Mean energies vs time -- semilogx'+'.eps',bbox_inches='tight')
    plt.close()

        
def bindingstid(hlim):
    plt.figure()#figsize=[4, 3.25])
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
# =============================================================================
#     fig1=plt.figure()
#     plt.plot(t,h,'ok',markersize=3)
#     plt.plot(tran,hcrit,'g:')
#     plt.xlabel('$t$')
#     plt.ylabel('$h$')
#     plt.savefig('hvt.eps')
#     plt.close()
# =============================================================================
    
# =============================================================================
#     fig1=plt.figure()
#     plt.loglog(t,h,'ok',markersize=3)
#     plt.loglog(tran,hcrit,'g:')
#     plt.xlabel('$t$')
#     plt.ylabel('$h$')
#     plt.savefig('hvt_loglog.eps')
#     plt.close()
# =============================================================================
    # =============================================================================
    #     fig1=plt.figure()
    #     plt.semilogy(t,h,'ok',markersize=3)
    #     plt.semilogy(tran,hcrit,'g:')
    #     plt.xlabel('$t$')
    #     plt.ylabel('$h$')
    #     plt.savefig('hvt_semilogy.eps')
    #     plt.close()
    # =============================================================================
    
    fig1=plt.figure()
    plt.semilogx(t,h,'ok',markersize=3)
    plt.semilogx(tran,hcrit,'g:')
    plt.xlabel('$t$')
    plt.ylabel('$h$')
    plt.savefig('hvt_semilogx.eps')
    plt.close()
    
    return tb1

def growthrate(t0,t1,Q):
    plt.figure()#figsize=[4, 3.25])
    inputdatas=np.load('sizefactors.npz')
    sizefactors=inputdatas['sizefactors']
    profts=inputdatas['profts']
    exponent=np.nan
    
    if sizefactors[-1]>0:
        Nt=len(profts)
        dt=profts[1]-profts[0]
        it0=round(t0/dt)
        it1=round(t1/dt)
        fit=np.polyfit(np.log(profts[it0:it1]), np.log(sizefactors[it0:it1]), 1)
        exponent=fit[0]
        fig1=plt.figure()
        ax=plt.axes()
        ax.plot(profts,sizefactors,'rx',label='Size factor')
        ax.plot(profts[it0:it1],(exp(fit[1])/(1))*profts[it0:it1]**exponent,'--b',linewidth=1.5,label='$\propto t^{'+str(round(exponent,2))+'}$')
        ax.set_xlabel('t')
        ax.set_ylabel('a')
        ax.set_title('Q = '+str(Q))
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend(loc='lower right')
        plt.savefig('sizefactorvt_logfit.eps',bbox_inches='tight')
        plt.close()
    return exponent

def NEWgrowthrate(t0,t1,Q):
    inputdatas=np.load('charks.npz')
    charks=inputdatas['charks']
    profts=inputdatas['profts']
    exponent=np.nan
    Nt=len(profts)
    dt=profts[1]-profts[0]
    it0=round(t0/dt)
    it1=round(t1/dt)
    fit=np.polyfit(np.log(profts[it0:it1]), np.log(charks[it0:it1]), 1)
    exponent=fit[0]
    #print('exponent='+str(exponent))
    fig1=plt.figure()
    ax=plt.axes()
    ax.plot(profts,charks,'rx',label='fft')
    #ax.plot(ts,kchar1s,'bx',label='fft')
    ax.plot(profts[it0:it1],(exp(fit[1])/(1))*profts[it0:it1]**exponent,'--b',linewidth=1.5,label='$\propto t^{'+str(round(exponent,2))+'}$')
    
    ax.set_xlabel('t')
    ax.set_ylabel('<k>')
    #ax.set_title('Q = '+str(Q))
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(loc='upper right')
    plt.savefig('kvt_logfit.eps',bbox_inches='tight')
    plt.close()
        
    #lamd1s=2*np.pi/kchar1s
    lamd2s=2*np.pi/charks
    
    fit=np.polyfit(np.log(profts[it0:it1]), np.log(lamd2s[it0:it1]), 1)
    exponent=fit[0]
    fig1=plt.figure()
    ax=plt.axes()
    ax.plot(profts,lamd2s,'rx',label='fft')
    #ax.plot(ts,lamd1s,'bx',label='fft')
    ax.plot(profts[it0:it1],(exp(fit[1])/(1))*profts[it0:it1]**exponent,'--b',linewidth=1.5,label='$\propto t^{'+str(round(exponent,2))+'}$')
        
    ax.set_xlabel('t')
    ax.set_ylabel('a')
    #ax.set_title('Q = '+str(Q))
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(loc='lower right')
    plt.savefig('NΕW_sizefactorvt_logfit.eps',bbox_inches='tight')
    plt.close()
    
    return exponent


def flukt_memb_TF(Q,B,h0,l1,l2,tau1off,sigmaon,x1,nx,profilinterval,profilyn,deltax,dt,t_max,hcut):
    starttime=time.time()
        
    Q=float(Q)
        
    B=float(B)
    tau2off=tau1off
    tau1on=tau1off/3
    tau2on=tau2off/3
    taur= tau1on/tau1off
        
    dt0=dt
    x0=0
    lr=l1/l2
        
    nt=int(t_max/(dt0*profilinterval))
    
    tau1on=float(tau1on)
    tau2on=float(tau2on)
    tau1off=float(tau1off)
    tau2off=float(tau2off)
    sigmaon=float(sigmaon)
    
    l1=float(l1)
    l2=float(l2)
    h0=float(h0)
    
    #% Lag mesh
    dt=Constant(dt)
    deltax=(x1-x0)/nx
    
    mesh = RectangleMesh( Point(x0,x0), Point(x1, x1), nx, nx, diagonal='right')
    #mesh = generate_mesh(domain)
    #plot(mesh)
    class Periodic_sides(SubDomain):
        def inside(self, x, on_boundary):
            return bool((near(x[0], 0) or near(x[1], 0)) and (not ((near(x[0],0) and near(x[1],x1)) or (near(x[0],x1) and near(x[1],0)))) and on_boundary)
        
        def map(self, x, y):
            if near(x[0],x1) and near(x[1],x1):
                y[0] = x[0] - x1
                y[1] = x[1] - x1
            elif near(x[0],x1):
                y[0] = x[0] - x1
                y[1] = x[1]
            elif near(x[1],x1):
                y[0] = x[0]
                y[1] = x[1] - x1
            else:
                y[0]=-1000
                y[1]=-1000  
        
        
    #% Define mixed function space
    EL = FiniteElement('P',triangle,1)
    element = MixedElement([EL,EL,EL])
    VWS = FunctionSpace(mesh,element, constrained_domain=Periodic_sides())
    # Define initial condition 
    hpm_0 = Expression(('h0','0','0'), degree=2,x1=x1,h0=h0)
    
    # Define trial and test functions within function space
    v, w,s = TestFunctions(VWS)
    hpm = Function(VWS)
    h,p,m, = split(hpm)
    # Initialize solution in our function space at t=0
    hpm_n=project(hpm_0, VWS)
    h_n, p_n,m_n = split(hpm_n)
    # Make eta a function
    elmts = VectorElement('Lagrange', mesh.ufl_cell(), 1)
    etaV=FunctionSpace(mesh,elmts)
    eta=Function(etaV)
    
    ## Formulate the variational problem
    # Write out weak form of the equations with all terms on one side
    #
    #Original form qith 1/12F=h_n*v*dx -h*v*dx-dt*Q*h**(3/2)*dot(grad(v),eta)*dx-dt*h**3*(1/(12))*dot(grad(p),grad(v))*dx +p*w*dx +B*dot(grad(w),grad(m))*dx +m*s*dx +dot(grad(s),grad(h))*dx   #+z1*c1_n*dx -z1*c1*dx +dt*z1*dot(grad(p),grad(c1))*(h*l1/mu)*dx +dt*(c10-c1)*z1*((1/tau1on)*exp(((l1-h)/(sigmaon*l1))**2))*dx -dt*c1*z1*((1/tau1off)*exp(((l1-h)/(sigmaoff*l1))**2))*dx -dt*D1*dot(grad(z1),grad(c1))*dx   +z2*c2_n*dx -z2*c2*dx +dt*z2*dot(grad(p),grad(c2))*(h*l2/mu)*dx +dt*(c20-c2)*z2*((1/tau2on)*exp(((l2-h)/(sigmaon*l2))**2))*dx -dt*c2*z2*((1/tau2off)*exp(((l2-h)/(sigmaoff*l2))**2))*dx -dt*D2*dot(grad(z2),grad(c2))*dx     
    F=h_n*v*dx -h*v*dx-dt*Q*h**(3/2)*dot(grad(v),eta)*dx-dt*h**3*dot(grad(p),grad(v))*dx +p*w*dx +B*dot(grad(w),grad(m))*dx +m*s*dx +dot(grad(s),grad(h))*dx   #+z1*c1_n*dx -z1*c1*dx +dt*z1*dot(grad(p),grad(c1))*(h*l1/mu)*dx +dt*(c10-c1)*z1*((1/tau1on)*exp(((l1-h)/(sigmaon*l1))**2))*dx -dt*c1*z1*((1/tau1off)*exp(((l1-h)/(sigmaoff*l1))**2))*dx -dt*D1*dot(grad(z1),grad(c1))*dx   +z2*c2_n*dx -z2*c2*dx +dt*z2*dot(grad(p),grad(c2))*(h*l2/mu)*dx +dt*(c20-c2)*z2*((1/tau2on)*exp(((l2-h)/(sigmaon*l2))**2))*dx -dt*c2*z2*((1/tau2off)*exp(((l2-h)/(sigmaoff*l2))**2))*dx -dt*D2*dot(grad(z2),grad(c2))*dx     
    #corrected form wo 1/12
    J=derivative(F,hpm)
    
    problem=NonlinearVariationalProblem(F,hpm,J=J)
    solver=NonlinearVariationalSolver(problem)
    
    #% lag filer der data kan lagres
    t_series=open('t.txt','w')   # create and open text files in which to write down parameter values
    t_profiles=open('t_profil.txt','w')
    h_series=open('h.txt','w')
    xdmffile = XDMFFile('solution.xdmf') #create xdmf file for the solution
    xdmffile.parameters["flush_output"] = True
    profiles=np.zeros((nx+1,nx+1,nt+1))
    ts=np.zeros(nt)
    xs=np.zeros(nx+1)
    ys=np.zeros(nx+1)
    profts=np.zeros((nt+1))
    x=mesh.coordinates()
    xs=x[0:nx+1,0]
    ys=x[0:nx+1,0]
    nxd=nx
    xd=np.linspace(x0,x1,num=nxd)
    min=1000
    max=1000
    xmin=0
    for i1 in range(nxd-1):
        for i2 in range(nxd-1):
            val=hpm_n(xd[i1],xd[i2])[0]
            if val < min:
                min=val
                idmx=i1
                idmy=i2
    xmin=xd[idmx]
    ymin=xd[idmy]
    h_series.write(str(round(min,6)))       # write value
    h_series.write('\n') 
    
    
    def stochastic(Number,deltax,dt):
        mesh = UnitIntervalMesh(Number)        # lag mesh
        elms = FiniteElement('Lagrange', mesh.ufl_cell(), 1)  #
        U = FunctionSpace(mesh, elms)     # funksjonsrom,
        u = TrialFunction(U)
        v = TestFunction(U)
        eta = Function(U)
        stdevinv = deltax*np.sqrt(dt)
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
        
    #% registrer metadata
    simpars=open('simparametere.txt','w')
    simpars.write('nx = '+str(nx))
    simpars.write('\n')
    simpars.write('x0 = '+str(x0))
    simpars.write('\n')
    simpars.write('x1 = '+str(x1))
    simpars.write('\n')
    simpars.write('dt = '+str(dt0))
    simpars.write('\n')
    simpars.write('Q = '+str(Q))
    simpars.write('\n')
    simpars.write('B = '+str(B))
    simpars.write('\n')
    simpars.write('h_0 = '+str(h0))
    simpars.write('\n')
    
    #% Initialisér løsning
    F0=h*v*dx-h_n*v*dx +p*w*dx-p_n*w*dx   +m*s*dx -m_n*s*dx 
    solve(F0==0,hpm)
        
    #% Loop i tid
    t=0
    profts[0]=t
    #xdmffile.write(_h,t)
    t_series.write(str(round(t,6)))
    t_series.write('\n')
    t_profiles.write(str(round(t,6)))
    t_profiles.write('\n')
    for ix in range(nx+1):
        for iy in range(nx+1):
            profiles[ix,iy,0]=h0
    itno_old =0
    i=0
    iprof=0
    imax=-1000
    vals = np.zeros(2)
    while min>hcut and t<t_max:
        i+=1
        # update current timestep
        t+=float(dt)                       # Time for which we will be computing solution
        
        print('Iteration number: '+str(i))#+'  /'+str(Nt))
        print('t = '+str(round(float(t),7)))#+'  /'+str(T))
        stochastic(len(eta.vector()),deltax,dt0)    # oppdater eta i tekstfil
        
        etaraw=np.loadtxt('u.txt') # last inn eta
        etaraw=etaraw[:,1]          # ta bare andre kolonne
        for ik in range(len(etaraw)-1):
            #print(ik)
            eta.vector()[ik] = etaraw[ik]         # sett vektorverdiene inn i funksjonen eta                         *********************
            if (np.sqrt(eta.vector()[ik]**2-etaraw[ik]**2)) != 0:
                print('Fail to assign new values to eta')
                
    
        # solve the variational problem to get new u at current timestep
        itno,conv =solver.solve()
            
        _h,_p, _m= hpm.split()
        
        hpm_n.assign(hpm)
        t_series.write(str(t))
        t_series.write('\n')
            
        
        if profilyn==1 and i/profilinterval==int(i/profilinterval):
            iprof+=1
            t_profiles.write(str(round(t,6)))
            t_profiles.write('\n')
            profts[iprof]=t
            for ix in range(nx+1):
                for iy in range(nx+1):
                    profiles[ix,iy,iprof]=_h(xs[ix],ys[iy])
            xdmffile.write(_h,t)
            
        min=1000
        for i1 in range(nxd-1):
            for i2 in range(nxd-1):
                val=hpm_n(xd[i1],xd[i2])[0]
                if val < min:
                    min=val
                    idmx=i1
                    idmy=i2
        h_series.write(str(round(min,6)))       # write value
        h_series.write('\n') 
        print('hmin = '+str(round(float(min),7)))#+'  /'+str(T))
    tick=0
    
    if t>0.99*t_max:
        t=np.nan
        tick=1
    #%
    h_series.close()
    t_series.close()
    t_profiles.close()
    
    endtime=time.time()
    print('Elapsed time = '+str(datetime.timedelta(seconds=endtime-starttime)))
    simpars.write('Elapsed real time = '+str(datetime.timedelta(seconds=endtime-starttime)))
    simpars.write('\n')
    simpars.write('# of iterations = '+str(i))
    simpars.write('\n')
    simpars.write('Simulation end time = '+str(t))
    simpars.close()
    
    #%
    np.savez('profiles.npz',profiles=profiles,xs=xs,ys=ys,profts=profts)
    np.savez('rupturetime.npz',tR=t)
    return tick


def gjennomsnittsprofil(N,h0,hcut,intormemb,x1):
    cwd = os.getcwd()
    
    
    
    
    