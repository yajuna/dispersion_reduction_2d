# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 08:54:56 2014

@author: yajun
"""

import numpy as np
import matplotlib.pyplot as plt

## test values
k = 0.75*np.pi
l = 2
M = 6
gamma = 0.3
phi0 = 0.25*np.pi # hope is to be close to zero angle
## 
config = dict()
config['iterNum'] = 37 
config['phiZeroEnd'] = 0.25*np.pi
config['tol'] = 2**(-4)




def remez2d(k = 0.5*np.pi, l = 4, M = 10, gamma = 0.3, phi0 = 0.25*np.pi, kfix = 0.75*np.pi, phifix = 0.25*np.pi, config1 = {'iterNum': 50, 'phiZeroEnd': np.pi*0.25, 'tol': 2**(-5)}):
#    k is defined to be a tuple
#    l the number of accuracy conditions at the zero wavenumber
#    M the total number of conditions
#    gamma = CFL number
#    phi0 is the angles that accuracy conditions are satisfied
#    print("gamma",gamma) 
    
    iterNum = config['iterNum'] = 50 
    tol = config['tol'] = 2**(-5)
#    print(iterNum, tol)
    
    top = 37*np.ones((l+1,M+2))
    for m in range(0,M+2):
        for j in range(0,l+1-2):
            top[j,m] = m**(2*j) * (np.cos(phi0)**(2*j) + np.sin(phi0)**(2*j))
        top[-2,m] = np.cos(m*kfix*np.cos(phifix)) + np.cos(m*kfix*np.sin(phifix))#exact phase at the point
        top[-1,m] = m*(np.cos(phifix)*np.sin(m*kfix*np.cos(phifix)) + np.sin(phifix)*np.sin(m*kfix*np.sin(phifix)))
# exact group velocity    
            
    top[:,M+1] = 0
#    print top[0,:]
    ##################################################
    phi = np.linspace(0.01,0.25*np.pi,M-l+1) # Due to periodicity
    # M-l+1 numbers, including the endpoints
    phi = np.array(phi).reshape(M-l+1,1)
    
    bottom = 37*np.ones((M-l+1,M+2))
    ##################################################
    cond = []
    for nn in range(1,iterNum):# main iterative step, does not converge within ~20 steps
        for j in range(0,M-l+1): #row
    
            for m in range(0,M+1): #column
                 bottom[j,m] = np.cos(m*k*np.cos(phi[j])) + np.cos(m*k*np.sin(phi[j]))
            bottom[j,M+1] = (-1)**j 
        LHS = np.vstack([top,bottom])
#        cond.append(np.linalg.cond(LHS))        
#        print ("condition number of the LHS",np.linalg.cond(LHS))
        RHS = []
        for j in range(0,l+1-2):
            RHS.append(-gamma**(2*j))
            
        RHS.append(-np.cos(gamma*kfix))
        RHS.append(-np.sin(gamma*kfix)*gamma)
        
        for j in range(0,M+2-(l+1)):
            RHS.append(-eval('fun(k,gamma)')) 
        
#        rhs = np.array(RHS).reshape(M+2,1)
        
        RHS = np.asarray(RHS)
        RHS = RHS.astype(np.float64) 
        rhs = RHS.reshape(M+2,1)
       
        c = np.linalg.solve(LHS,rhs)
#        resi=np.dot(LHS,c)-rhs
#        print ("max error of residue",np.max(np.abs(resi)))
#        solve for the coeff and error
        c1 = c[0:M+1]# only coefficients
    ##################################################
        phizero = 73*np.ones((M-l+2,1))
        phizero[0] = 0
        phizero[M-l+1] = config['phiZeroEnd']
       
        for j in range(1,M-l+1):
            phizero[j] = findzero(err,phi[j-1],phi[j],fun,c,M,k,gamma) #??!!!??!!   
            
        phi1 = 373*np.ones((M-l+1,1))
        
        v = 737*np.ones((M-l+1,1))
        for j in range(1,M-l+2):
            if np.sign(err_der(phizero[j-1],fun_der,c,M,k,gamma)) != np.sign(err_der(phizero[j],fun_der,c,M,k,gamma)):
                phi1[j-1] = findzero(err_der,phizero[j-1],phizero[j],fun_der,c,M,k,gamma)
                v[j-1] = np.abs(err(phi1[j-1],fun,c,M,k,gamma))
            
            else:
                v1 = np.abs(err(phizero[j-1],fun,c,M,k,gamma))
                v2 = np.abs(err(phizero[j],fun,c,M,k,gamma))
                if v1>v2:
                    phi1[j-1] = phizero[j-1]
                    v[j-1] = v1
                else:
                    phi1[j-1] = phizero[j]
                    v[j-1] = v2 
        ind_max = np.argmax(v)#v[ind_max] gives the max value
    
        
        if np.abs(phi[ind_max]-phi1[ind_max])<tol:
            return c1#,phi
        
        elif ind_max+1<np.size(phi) and np.abs(phi[ind_max+1]-phi1[ind_max])<tol: 
            return c1#,phi
        
        
        else:
            
            piancha = np.ones((M-l+1,1))
            for j in range(0,M-l+1):
                piancha[j] = err(phi1[j],fun,c,M,k,gamma)
            dapiancha = np.max(np.abs(piancha))
            
            print ("print error for each iteration",dapiancha)
#            print("current coeff with above error is c1 = ", c1)            
            phi = phi1
#            print ("updated reference",phi1)
            
#########################################################
# define the zero finding function. 
def findzero(fun,x0,x1,*args):
    
    # zeros between x0 and x1
    #f0=0
    #f1=0
   
    f0 = eval('fun(x0,*args)')
    #print ("f0",f0)
    f1 = eval('fun(x1,*args)')
    #print ("f1",f1)
    if np.sign(f0)==np.sign(f1):
        print("function at the two end points of same sign")
        
    x = (x0+x1)/2.0 #x0-f0*(x1-x0)/(f1-f0)
    #print ("x",x)
    #time.sleep(10)
    f = eval('fun(x,*args)')
    #print ("f",f)    
    while np.abs(f)>2**(-10):# if 2^-30 to 2^-10, max error does not decrease monot
        if np.sign(f)==np.sign(f0):
            x0 = x
            f0 = f
        else:
            x1 = x
            f1 = f
        x = (x0+x1)/2.0 #x0-f0*(x1-x0)/(f1-f0)
        f = eval('fun(x,*args)')
           
   
    return x
#########################################################
# define the error function
def err(phi,fun,c,M,k,*args):
     # difine the error function
    c1 = np.array(c[0:M+1]).reshape(1,M+1)# only coeff
    
    
    d = []
    
    for m in range(0,M+1):
        d.append(np.cos(m*k*np.cos(phi)) + np.cos(m*k*np.sin(phi)))
    #print("shape of d",np.shape(d))
    d = np.array(d).reshape(M+1,1) 
    ################################
#    d=np.ones((M+1,1))
#    for j in range(0,M+1):
#        d[j]=np.cos(j*k)
    #################################
    #print ("line 136 gives the shape of d",np.shape(d))    
    error = eval('fun(k,*args)') + np.dot(c1,d)
    
    
    return error
##############
# define the derivative of error function
def err_der(phi,fun_der,c,M,k,*args):
     # difine the error function
    c1 = np.array(c[0:M+1]).reshape(1,M+1)# only coeff
    
    d1 = []
    for m in range(0,M+1):
        d1.append(np.sin(m*k*np.cos(phi))*m*k*np.sin(phi) - np.sin(m*k*np.sin(phi))*m*k*np.cos(phi))
    d1 = np.array(d1).reshape(M+1,1)     
    error_der = eval('fun_der(k,*args)') + np.dot(c1,d1)
    
    return error_der
    ##############
          
#########################################################
def fun(K,gamma):
    return np.cos(gamma*K)
#########################################################
def fun_der(K,gamma):
    return 0# since the true dispersion is angle independent               
            

#k=0.2*np.pi
#polar=1
#l=3
#M=14
#gamma=0.6
#phi0=0.25*np.pi
#
#def remez2dgroup(k,l,M,gamma,phi0,polar):
##    k is defined to be a tuple
##    l the number of accuracy conditions at the zero wavenumber
##    M the total number of conditions
##    gamma=CFL number
##    phi0 is picked to be angles that accuracy conditions are satisfied
#
#    top=37*np.ones((l+1,M+2))
#    for m in range(0,M+2):    
#        for j in range(0,l-2):
#            top[j,m]=m**(2*j)*(np.cos(phi0)**(2*j)+np.sin(phi0)**(2*j))
#        if polar==1: 
#            
#            top[-3,m]=np.cos(m*k*np.cos(phi0))+np.cos(m*k*np.sin(phi0))#exact phase at the point
#            top[-2,m]=m*(np.cos(phi0)*np.sin(m*k*np.cos(phi0))+np.sin(phi0)*np.sin(m*k*np.sin(phi0)))
#            top[-1,m]=m*k*(np.sin(phi0)*np.sin(m*k*np.cos(phi0))+np.cos(phi0)*np.sin(m*k*np.sin(phi0)))
##        if polar==0:
#            
#    top[:,M+1]=0
##    print ("top part of matrix",top)
#    ##################################################
#    phi=np.linspace(0,0.5*np.pi,M-l+1) # Due to periodicity, change !
#    phi=np.array(phi).reshape(M-l+1,1)
#    
#    bottom=37*np.ones((M-l+1,M+2))
#    ##################################################
#    cond=[]
#    for nn in range(1,50):# main iterative step, does not converge within ~20 steps
#        for j in range(0,M-l+1): #row
#    
#            for m in range(0,M+1): #column
#                 bottom[j,m]=np.cos(m*k*np.cos(phi[j]))+np.cos(m*k*np.sin(phi[j]))
#        bottom[j,M+1]=(-1)**j 
#        LHS=np.vstack([top,bottom])## singular!!??###???!?!?!?!
#        cond.append(np.linalg.cond(LHS))        
#        print ("condition number of the LHS",np.linalg.cond(LHS))
#        RHS=[]
#        for j in range(0,l-2):
#            RHS.append(-gamma**(2*j))
#        if polar==1: 
#            RHS.append(-np.cos(gamma*k))
#            RHS.append(-gamma*np.sin(gamma*k))
#            RHS.append(0)
#        for j in range(0,M+2-(l+1)):
#            RHS.append(-eval('fun(k,gamma)')) 
#        
#        rhs=np.array(RHS).reshape(M+2,1)
#        #print("rhs",rhs)
#       
#        c=np.linalg.solve(LHS,rhs)
##        resi=np.dot(LHS,c)-rhs
##        print ("max error of residue",np.max(np.abs(resi)))
##        solve for the coeff and error
#        c1=c[0:M+1]# only coefficients
#    ##################################################
#        phizero=73*np.ones((M-l+2,1))
#        phizero[0]=0
#        phizero[M-l+1]=np.pi*2.0 #If np.pi*1.8, 0.9pi and 0.125pi does not converge??!!
#       
#        for j in range(1,M-l+1):
#            phizero[j]=findzero(err,phi[j-1],phi[j],fun,c,M,k,gamma) #??!!!??!!   
#            
#        phi1=373*np.ones((M-l+1,1))
#        
#        v=737*np.ones((M-l+1,1))
#        for j in range(1,M-l+2):
#            if np.sign(err_der(phizero[j-1],fun_der,c,M,k,gamma))!=np.sign(err_der(phizero[j],fun_der,c,M,k,gamma)):
#                phi1[j-1]=findzero(err_der,phizero[j-1],phizero[j],fun_der,c,M,k,gamma)
#                v[j-1]=np.abs(err(phi1[j-1],fun,c,M,k,gamma))
#            
#            else:
#                v1=np.abs(err(phizero[j-1],fun,c,M,k,gamma))
#                v2=np.abs(err(phizero[j],fun,c,M,k,gamma))
#                if v1>v2:
#                    phi1[j-1]=phizero[j-1]
#                    v[j-1]=v1
#                else:
#                    phi1[j-1]=phizero[j]
#                    v[j-1]=v2 
#        ind_max=np.argmax(v)#v[ind_max] gives the max value
#    
#        tol=2**(-3)# changing tol from 2^-30 to 2^-10 decreased the max error
#        if np.abs(phi[ind_max]-phi1[ind_max])<tol:
#            return c1#,phi
#        
#        elif ind_max+1<np.size(phi) and np.abs(phi[ind_max+1]-phi1[ind_max])<tol: 
#            return c1#,phi
#        
#        
#        else:
#            
#            piancha=np.ones((M-l+1,1))
#            for j in range(0,M-l+1):
#                piancha[j]=err(phi1[j],fun,c,M,k,gamma)
#            dapiancha=np.max(np.abs(piancha))
#            
#            print ("print error for each iteration",dapiancha)            
#            phi=phi1
#            print ("updated reference",phi1)
##########################################################
## define the zero finding function. 
#def findzero(fun,x0,x1,*args):
#    
#    # zeros between x0 and x1
#    #f0=0
#    #f1=0
#   
#    f0=eval('fun(x0,*args)')
#    #print ("f0",f0)
#    f1=eval('fun(x1,*args)')
#    #print ("f1",f1)
#    if np.sign(f0)==np.sign(f1):
#        print ("function at the two end points of same sign")
#        
#    x=(x0+x1)/2.0 #x0-f0*(x1-x0)/(f1-f0)
#    #print ("x",x)
#    #time.sleep(10)
#    f=eval('fun(x,*args)')
#    #print ("f",f)    
#    while np.abs(f)>2**(-10):# if 2^-30 to 2^-10, max error does not decrease monot
#        if np.sign(f)==np.sign(f0):
#            x0=x
#            f0=f
#        else:
#            x1=x
#            f1=f
#        x=(x0+x1)/2.0 #x0-f0*(x1-x0)/(f1-f0)
#        f=eval('fun(x,*args)')
#           
#   
#    return x
##########################################################
## define the error function
#def err(phi,fun,c,M,k,*args):
#     # difine the error function
#    c1=np.array(c[0:M+1]).reshape(1,M+1)# only coeff
#    
#    
#    d=[]
#    
#    for m in range(0,M+1):
#        d.append(np.cos(m*k*np.cos(phi))+np.cos(m*k*np.sin(phi)))
#    #print("shape of d",np.shape(d))
#    d=np.array(d).reshape(M+1,1) 
#    ################################
##    d=np.ones((M+1,1))
##    for j in range(0,M+1):
##        d[j]=np.cos(j*k)
#    #################################
#    #print ("line 136 gives the shape of d",np.shape(d))    
#    error=eval('fun(k,*args)')+np.dot(c1,d)
#    
#    
#    return error
###############
## define the derivative of error function
#def err_der(phi,fun_der,c,M,k,*args):
#     # difine the error function
#    c1=np.array(c[0:M+1]).reshape(1,M+1)# only coeff
#    
#    d1=[]
#    for m in range(0,M+1):
#        d1.append(np.sin(m*k*np.cos(k))*m*k*np.sin(phi)-np.sin(m*k*np.sin(k))*m*k*np.cos(phi))
#    d1=np.array(d1).reshape(M+1,1)     
#    error_der=eval('fun_der(k,*args)')+np.dot(c1,d1)
#    
#    return error_der
#    ##############
#          
##########################################################
#def fun(K,gamma):
#    return np.cos(gamma*K)
##########################################################
#def fun_der(K,gamma):
#    return 0# since the true dispersion is angle independent               
#            