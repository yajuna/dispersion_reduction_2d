#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 11:32:26 2019

@author: yajun

computes stability factors for 2D schemes
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def stafac(soln):#x,y->phi,k
    phi = np.linspace(0, np.pi/2.0, 100)#include bdry values
    k = np.linspace(0,np.pi,100)
#    X, Y = np.meshgrid(phi, k)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(phi, k)
    zs = np.array([foo(soln, phi, k) for phi,k in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('$\phi$ propagation angle')
    ax.set_ylabel('k wavenumber')
    ax.set_zlabel('stability factor')

    print ('Stability factor, max = %.16f, min=%.16f'%(Z.max(), Z.min()))
    
    return np.max(np.abs(Z))

# phi and k are vectors ranging (0,2pi) and (0,pi) respectively
def foo(coeff,phi,k):#x,y->phi,k
    M = np.size(coeff)
    _sum = 0
    for m in range(M):
        _sum = _sum + coeff[m]*(np.cos(m*k*np.cos(phi)) + np.cos(m*k*np.sin(phi)))
    #print("Maximum for factor",np.max(np.abs(he)))
    return _sum   

######### optional visualization
