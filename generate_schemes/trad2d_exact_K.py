# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:25:31 2015

This code computes coefficients for 2D isotropic wave equation, with accuracy 
conditions at the zero wavenumber at angles phi_0, and exact dispersion at
wavenumbers K=[] along the angle phi

@author: yajun
"""



import numpy as np
import matplotlib.pyplot as plt

config = dict()

config['phi0'] = 0.25*np.pi
config['phi'] = 0.25*np.pi
config['gamma'] = 0.6
config['M'] = 10
config['K'] = np.linspace(0.05*np.pi,0.9*np.pi,8)

def trad2d_exact(config):
    phi0 = config['phi0']
    gamma = config['gamma'] 
    M = config['M'] 
    phi = config['phi']
    K = config['K']
    l = M-np.size(K)#l + 1 + np.size(K) = M + 1
    print('l = ',l)
    matrix = 37*np.ones((M+1,M+1))
    rhs = 73*np.ones((M+1,1))
    for j in range(0,M+1):
        for m in range(0,M+1):
            matrix[j,m] = m**(2*j)*(np.cos(phi0)**(2*j) + np.sin(phi0)**(2*j))
        rhs[j] = -gamma**(2*j)
    for j in range(l+1,M+1):
        for m in range(0,M+1):
            matrix[j,m] = np.cos(m*K[j-l-1]*np.cos(phi)) + np.cos(m*K[j-l-1]*np.sin(phi))
        rhs[j] = -np.cos(gamma*K[j-l-1])
    coeff = 373*np.ones((M+1,1))
#    print("shape of matrix and rhs",np.shape(matrix),np.shape(rhs))
    coeff = np.linalg.solve(matrix,rhs)
    return coeff
    
c = trad2d_exact(config)

print('coefficient is ', c)
    
# phi and k are vectors ranging (0,2pi) and (0,pi) respectively
def foo(c,phi,k):#x,y->phi,k
    M = np.size(c)
    he = 0
    for m in range(M):
        he = he + c[m]*(np.cos(m*k*np.cos(phi)) + np.cos(m*k*np.sin(phi)))
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

ax.set_xlabel('Propagation angle phi')
ax.set_ylabel('Wavenumber k')
ax.set_zlabel('Stability factor')

print ('Stability factor, max = %.16f, min=%.16f'%(Z.max(), Z.min()))