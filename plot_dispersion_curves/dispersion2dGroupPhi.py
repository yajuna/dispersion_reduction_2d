#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 19:42:55 2021

Similar to dispersionCurve2dGroup, but plot dispersion error for all angle phi instead
of wavenumber K

@author: yajun
"""

import numpy as np
import matplotlib.pyplot as plt

###################### parameter
epsilon = 0.0

phimin = np.pi*0.0
phimax = np.pi
num = 5000

gamma = 0.3

c = np.array([[-3.59892052e-01],
       [-1.68134419e-01],
       [ 3.90164665e-02],
       [-1.56535484e-02],
       [ 6.66223090e-03],
       [-2.81799300e-03],
       [ 1.08890297e-03],
       [-3.34732907e-04],
       [ 7.57973126e-05],
       [-1.15872858e-05],
       [ 9.35674437e-07]])

M = np.size(c)-1

############## draw the dispersion curve
def dispCurve2d(gamma, c, k):
    phi = np.linspace(epsilon,phimax,num)
    phi = np.array(phi).reshape(num,1)
    ai = np.ones((num,1))
# For generalized leap-frog, _sum is a vector of shape(num,1)
    _sum1 = np.zeros((num,1))
    for m in range(0,M+1):
        _sum1 = _sum1 + c[m]*(np.cos(m*k*np.cos(phi)) + np.cos(m*k*np.sin(phi)))
    SquareSum1 = _sum1**2
    
    _sum2 = np.zeros((num,1))   
    for m in range(0,M+1):
        _sum2 = _sum2 + c[m]*(np.sin(m*k*np.cos(phi))*m*np.cos(phi) + np.sin(m*k*np.sin(phi))*m*np.sin(phi))
    SquareSum2 = _sum2**2
    
    _sum3 = np.zeros((num,1))   
    for m in range(0,M+1):
        _sum3 = _sum3 + c[m]*(-np.sin(m*k*np.cos(phi))*m*k*np.sin(phi) + np.sin(m*k*np.sin(phi))*m*k*np.cos(phi))
    SquareSum3 = _sum3**2
    
        # magnitude of nabla Omega
#    dispersionArray = ai / (gamma * np.sqrt(ai-SquareSum1)) * np.sqrt(SquareSum2 + SquareSum3)
    
    # partial Omega/partial K
    dispersionArray = -ai / (gamma * np.sqrt(ai-SquareSum1)) * _sum2#np.sqrt(SquareSum2 + SquareSum3)
  
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
plt.ylabel('$c_{gFD}/c_g$')
plt.xlabel('$\phi$')

plt.savefig('/home/yajun/Documents/wave2d/figs/remez2dk075pi_025pi03gamma2l10mCurvePhiGroup1.eps', format='eps', dpi=300)
plt.show()

