# -*- coding: utf-8 -*-
"""
Created on Mon May  4 16:44:24 2015

This code computes schemes for the 2D wave equation. With fixed angle, and 
preset wavenumbers, we do Remez algorithm in wavenumbers. Stability factor is 
computed in the code stability_factor.py

phi = 0.25*np.pi, kfix = [], l = 2, M = 10, gamma = 0.3, phi0 = 0.125*np.pi, config = {'iterNum': 50, 'kEnd': 0.9*np.pi, 'kRange': 0.5*np.pi, 'tol': 2**(-3)})

Code very sensitive with respect to kRange. kRange = 0.5*np.pi gives bogus coefficients; kRange = 0.8*np.pi is fine (as in remez2d_may5); just initial 
reference points, should not matter at all.

example parameters:

phi = 0.25*np.pi    
k=0.5*np.pi
kfix=[0.3*np.pi] 
l=3
M=6
gamma=0.6
phi0=0.25*np.pi

@author: yajun
"""


    


import numpy as np
import matplotlib.pyplot as plt

#phi = 0.25*np.pi # fixed propagation angle at pi/4. 
#kfix = [] # preset wavenumbers
#l = 2
#M = 10
#gamma = 0.6
#phi0 = 0.25*np.pi # accuracy condition, can change to at n*pi/4, with n = 1,3,5,7

def remez2dphase(phi = 0.125*np.pi, kfix = [], l = 2, M = 10, gamma = 0.3, phi0 = 0.125*np.pi, config = {'iterNum': 50, 'kEnd': 0.9*np.pi, 'kRange': 0.8*np.pi, 'tol': 2**(-3)}):

    iterNum = config['iterNum'] 
    tol = config['tol'] 
    kEnd = config['kEnd']
    kRange = config['kRange']
    
    dim = (M+2)-(l+1)-np.size(kfix)
    top1 = 37*np.ones((l+1,M+2))
    for j in range(0,l+1):
        for m in range(0,M+2):
            top1[j,m] = m**(2*j)*(np.cos(phi0)**(2*j) + np.sin(phi0)**(2*j))
    top1[:,M+1] = 0
    
    top2 = np.zeros((np.size(kfix),M+2))
    for j in range(0,np.size(kfix)):# index up to np.size(kfix)-1
        for m in range(0,M+1):
            top2[j,m] = np.cos(m*kfix[j]*np.cos(phi0))+np.cos(m*kfix[j]*np.sin(phi0))
    top = np.vstack([top1,top2])
#    print ("top part of matrix",top)
    ##################################################
    k = np.linspace(0.01,kRange,dim) # Due to periodicity
    k = np.array(k).reshape(dim,1)
    
    bottom = 37*np.ones((dim,M+2))
    ##################################################
    cond = []
    for nn in range(1,iterNum):# main iterative step, does not converge within ~20 steps
        for j in range(0,dim): #row
    
            for m in range(0,M+1): #column
                 bottom[j,m] = np.cos(m*k[j]*np.cos(phi)) + np.cos(m*k[j]*np.sin(phi))
            bottom[j,M+1] = (-1)**j 
        LHS = np.vstack([top,bottom])
        cond.append(np.linalg.cond(LHS))        
        print ("condition number of the LHS",np.linalg.cond(LHS))
        RHS = []
        
        for j in range(0,l+1):
            RHS.append(-gamma**(2*j))
        for j in range(0,np.size(kfix)):
            RHS.append(-eval('fun(kfix[j],gamma)'))
        for j in range(0,dim):
            RHS.append(-eval('fun(k[j],gamma)'))##
#        print(type(RHS),np.size(RHS))
        RHS = np.asarray(RHS)
        RHS = RHS.astype(np.float64) 
        rhs = RHS.reshape(M+2,1)
        
       
        c = np.linalg.solve(LHS,rhs)

        c1 = c[0:M+1]# only coefficients
    ##################################################
        kzero = 73*np.ones((dim+1,1))
        kzero[0] = 0
        kzero[dim] = kEnd 
       
        for j in range(1,dim):
            kzero[j] = findzero(err,k[j-1],k[j],fun,c,M,phi,gamma)    
            
        k1 = 373*np.ones((dim,1))
        
        v = 737*np.ones((dim,1))
        for j in range(1,dim+1):
            if np.sign(err_der(kzero[j-1],fun_der,c,M,phi,gamma)) != np.sign(err_der(kzero[j],fun_der,c,M,phi,gamma)):
                k1[j-1] = findzero(err_der,kzero[j-1],kzero[j],fun_der,c,M,phi,gamma)
                v[j-1] = np.abs(err(k1[j-1],fun,c,M,phi,gamma))
            
            else:
                v1 = np.abs(err(kzero[j-1],fun,c,M,phi,gamma))
                v2 = np.abs(err(kzero[j],fun,c,M,phi,gamma))
                if v1>v2:
                    k1[j-1] = kzero[j-1]
                    v[j-1] = v1
                else:
                    k1[j-1] = kzero[j]
                    v[j-1] = v2 
        ind_max = np.argmax(v)#v[ind_max] gives the max value
    
        # changing tol from 2^-30 to 2^-10 decreased the max error
        if np.abs(k[ind_max]-k1[ind_max])<tol:
            return c1#,phi
        
        elif ind_max+1<np.size(k) and np.abs(k[ind_max+1]-k1[ind_max])<tol: 
            return c1#,phi
        
        
        else:
            
            piancha = np.ones((dim,1))
            for j in range(0,dim):
                piancha[j] = err(k1[j],fun,c,M,phi,gamma)
            dapiancha = np.max(np.abs(piancha))
            
            print ("print error for each iteration",dapiancha)            
            k = k1
            print ("updated reference",k1)
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
    if np.sign(f0) == np.sign(f1):
        print("function at the two end points of same sign")
        
    x = (x0+x1)/2.0 #x0-f0*(x1-x0)/(f1-f0)
    #print ("x",x)
    #time.sleep(10)
    f = eval('fun(x,*args)')
    #print ("f",f)    
    while np.abs(f)>2**(-10):# if 2^-30 to 2^-10, max error does not decrease monot
        if np.sign(f) == np.sign(f0):
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
def err(k,fun,c,M,phi,*args):
     # define the error function
    c1 = np.array(c[0:M+1]).reshape(1,M+1)# only coeff
    
    
    d = []
    
    for m in range(0,M+1):
        d.append(np.cos(m*k*np.cos(phi)) + np.cos(m*k*np.sin(phi)))
    
    d = np.array(d).reshape(M+1,1) 
    ################################
#    d=np.ones((M+1,1))
#    for j in range(0,M+1):
#        d[j]=np.cos(j*k)
    #################################
       
    error = eval('fun(k,*args)') + np.dot(c1,d)
    
    
    return error
##############
# define the derivative of error function
def err_der(k,fun_der,c,M,phi,*args):
     # define the error function
    c1 = np.array(c[0:M+1]).reshape(1,M+1)# only coeff
    
    d1 = []
    for m in range(0,M+1):
        d1.append(-np.sin(m*k*np.cos(phi))*m*np.cos(phi)-np.sin(m*k*np.sin(phi))*m*np.sin(phi))# may5: fix
    d1 = np.array(d1).reshape(M+1,1)     
    error_der = eval('fun_der(k,*args)') + np.dot(c1,d1)
    
    return error_der
    ##############
          
#########################################################
def fun(K,gamma):
    return np.cos(gamma*K)
#########################################################
def fun_der(K,gamma):
    return -gamma*np.sin(gamma*K)              
            
    #########################
    
    # check stability

#c = remez2dphase() 
#
#print("the coefficients for system with default values", c)
#
##############################################################################
#
## phi and k are vectors ranging (0,2pi) and (0,pi) respectively
#def foo(c,phi,k):#x,y->phi,k
#    M = np.size(c)
#    he = 0
#    for m in range(M):
#        he = he + c[m]*(np.cos(m*k*np.cos(phi))+np.cos(m*k*np.sin(phi)))
#    #print("Maximum for factor",np.max(np.abs(he)))
#    return he
#    
################################    
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#
#
#phi = np.linspace(0, np.pi/2.0, 100)#include bdry values
#k = np.linspace(0,np.pi,100)
#X, Y = np.meshgrid(phi, k)
#zs = np.array([foo(c,phi,k) for phi,k in zip(np.ravel(X), np.ravel(Y))])
#Z = zs.reshape(X.shape)
#
#ax.plot_surface(X, Y, Z)
#
#ax.set_xlabel('$\phi$ propagation angle')
#ax.set_ylabel('k wavenumber')
#ax.set_zlabel('stability factor')
#
#print ('Stability factor, max = %.16f, min=%.16f'%(Z.max(), Z.min()))
#
##plt.show()
