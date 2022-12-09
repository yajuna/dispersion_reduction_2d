# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 14:53:15 2015

This code gives coefficient computed for 2D wave equation.  
The equations are obtained by consistency condition, accuracy of dispersion 
relation, and exact dispersion at some angles, phi=[]
The code takes five input: phi0 the fixed propagation angle for accuracy; gamma
the CFL number; M the stencil width, phi the preset angles and K the wavenumber

@author: yajun
"""

import numpy as np
import matplotlib.pyplot as plt



config = dict()

config['phi0'] = 0.25*np.pi
config['gamma'] = 0.3
config['M'] = 10
config['phi'] = [0.25*np.pi,0.125*np.pi] 
config['K'] = 0.75*np.pi
def trad2d_exact(config):
    phi0 = config['phi0']
    gamma = config['gamma'] 
    M = config['M'] 
    phi = config['phi']
    K = config['K']
    l = M - np.size(phi)#l+1+np.size(c)=M+1
    print('l=',l)
    matrix = 37*np.ones((M+1,M+1))
    rhs = 73*np.ones((M+1,1))
    for j in range(0,M+1):
        for m in range(0,M+1):
            matrix[j,m] = m**(2*j)*(np.cos(phi0)**(2*j) + np.sin(phi0)**(2*j))
        rhs[j] = -gamma**(2*j)
    for j in range(l+1,M+1):
        for m in range(0,M+1):
            matrix[j,m] = np.cos(m*K*np.cos(phi[j-l-1])) + np.cos(m*K*np.sin(phi[j-l-1]))
        rhs[j] = -np.cos(gamma*K)
    coeff = 373*np.ones((M+1,1))
#    print("shape of matrix and rhs",np.shape(matrix),np.shape(rhs))
    coeff = np.linalg.solve(matrix,rhs)
    return coeff


c = trad2d_exact(config)

print (c, 'flag')
    
# phi and k are vectors ranging (0,2pi) and (0,pi) respectively
def foo(c,phi,k):#x,y->phi,k
    M = np.size(c)
    he = 0
    for m in range(M):
        he = he+c[m]*(np.cos(m*k*np.cos(phi)) + np.cos(m*k*np.sin(phi)))
    #print("Maximum for factor",np.max(np.abs(he)))
    return he
    
###############################    
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')



phi = np.linspace(0, np.pi/2.0, 100)#include bdry values
k = np.linspace(0,np.pi,100)
X, Y = np.meshgrid(phi, k)
zs = np.array([foo(c,phi,k) for phi,k in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

ax.plot_surface(X, Y, Z)

ax.set_xlabel('phi Label')
ax.set_ylabel('k Label')
ax.set_zlabel('factor Label')

print ('Stability factor, max = %.16f, min=%.16f'%(Z.max(), Z.min()))