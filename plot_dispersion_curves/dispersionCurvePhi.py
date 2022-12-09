#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 22:51:32 2020

Similar to dispersionCurve2d, but plot dispersion error for all angle phi instead
of wavenumber K

@author: yajun
"""
import numpy as np
import matplotlib.pyplot as plt

###################### parameter
epsilon = 0

phimin = np.pi*0.0
phimax = np.pi
num = 5000

gamma = 0.3

c = np.array([[-3.64078498e-01],
       [-1.59452112e-01],
       [ 3.11597693e-02],
       [-1.05561217e-02],
       [ 4.07359923e-03],
       [-1.56893278e-03],
       [ 5.58535169e-04],
       [-1.71578792e-04],
       [ 4.17672990e-05],
       [-7.02955736e-06],
       [ 6.02441602e-07]])

M = np.size(c)-1

############## draw the dispersion curve
def dispCurve2d(gamma, c, k):
    phi = np.linspace(epsilon,phimax,num)
    phi = np.array(phi).reshape(num,1)
    ai = np.ones((num,1))
# For generalized leap-frog, _sum is a vector of shape(num,1)
    _sum = np.zeros((num,1))
    for m in range(0,M+1):
        _sum = _sum + c[m]*(np.cos(m*k*np.cos(phi)) + np.cos(m*k*np.sin(phi)))
    
    dispersionArray = ai / (gamma*k)*np.arccos(-_sum)

    dispersionArray[0] = 1
    print ("max error for k =", k, "is", np.max(np.abs(dispersionArray - ai)))
    return dispersionArray

k = np.linspace(0.01,np.pi*0.8, 5)
l = []
for j in range(5):
    l.append(dispCurve2d(gamma, c, k[j]))
    
ai = np.ones((num,1))
phi = np.linspace(epsilon,phimax,num)
phi = np.array(phi).reshape(num,1)

line1 = plt.plot(phi,ai,'k',label="Exact") #
line2 = plt.plot(phi,l[0],'b',label="$k = 0.01$") #
line3 = plt.plot(phi,l[1],'g',label="$k = 0.2\pi$") #
line4 = plt.plot(phi,l[2],'r',label="$k = 0.4\pi$") #
line5 = plt.plot(phi,l[3],'c',label="$k = 0.6\pi$") #
line6 = plt.plot(phi,l[4],'m',label="$k = 0.8\pi$") #

#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc = 3,
#           ncol = 3, mode="expand", borderaxespad = -0.2)
plt.legend(loc = 'lower left')
plt.axis([epsilon,phimax,0.75,1.1])
plt.ylabel('$c_{FD}/c$')
plt.xlabel('$\phi$')

#plt.savefig('/home/yajun/Documents/wave2d/figs/remez2dgroup_025pi03gamma2l10mCurvePhi.eps', format='eps', dpi=300)
plt.show()

#%% compute max dispersion error

def MaxDispersion2d(gamma, c):
    
    M = c.size
    
    K = np.linspace(0, np.pi, 1000)
    phi = np.linspace(0, np.pi, 1001)
    kk, phiphi = np.meshgrid(K, phi, indexing = 'ij') # matrix indexing
    
    numDispersion = 0
    for m in range(M):
        numDispersion = + c[m] * (np.cos(m * kk * np.cos(phiphi)) + np.cos(m * kk * np.sin(phiphi)))
        
    
    error = np.cos(gamma * kk) + numDispersion 

    errorMax = error.max()
    
    return errorMax
