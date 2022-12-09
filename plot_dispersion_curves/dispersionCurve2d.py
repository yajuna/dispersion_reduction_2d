#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 16:45:19 2020

Plot dispersion curve and return max of dispersion error

Plot c = 1 = Omega_num/gamma*K = np.arccos(-sum^M_0 cm (cos(m K cos phi) + cos(m K sin phi)))/gamma * K

input: gamma, c_m, phi

modified from draw_dispersion_5th_ani.py
#######################################
Output max dispersion error as well

Function is of K and phi. Compute the following 

cos(Omega_exact) - cos(Omega_num) = 
cos(gamma*K) + sum^M_0 c_m*[cos(m*K*cos(phi)) + cos(m*K*sin(phi))]
                                    
input: gamma, c_m
parameters: K, phi as arrays

@author: yajun
"""
import numpy as np
import matplotlib.pyplot as plt

###################### parameter
epsilon = 0.01
gamma = 0.3

kmin = np.pi*0.0
kmax = np.pi

num = 5000


c = np.array([[-3.06767651e-01],
       [-2.68775340e-01],
       [ 1.25973609e-01],
       [-8.57946260e-02],
       [ 5.87616191e-02],
       [-3.68279583e-02],
       [ 1.96974533e-02],
       [-8.47976268e-03],
       [ 2.73077439e-03],
       [-5.78305640e-04],
       [ 6.01883813e-05]])

M = np.size(c)-1

############## draw the dispersion curve
def dispCurve2d(gamma, c, phi):
    k = np.linspace(epsilon,kmax,num)
    k = np.array(k).reshape(num,1)
    ai = np.ones((num,1))
# For generalized leap-frog, _sum is a vector of shape(num,1)
    _sum = np.zeros((num,1))
    for m in range(0,M+1):
        _sum = _sum + c[m]*(np.cos(m*k*np.cos(phi)) + np.cos(m*k*np.sin(phi)))
    
    dispersionArray = ai / (gamma*k)*np.arccos(-_sum)

    dispersionArray[0] = 1
    
    print ("max error for phi =", phi, "is", np.max(np.abs(dispersionArray - ai)))
    return dispersionArray

phi = np.linspace(0,np.pi/4, 5)
l = []
for j in range(5):
    l.append(dispCurve2d(gamma,c,phi[j]))
    

ai = np.ones((num,1))
k = np.linspace(epsilon,kmax,num)
k = np.array(k).reshape(num,1)

line1 = plt.plot(k,ai,'k',label="Exact") #
line2 = plt.plot(k,l[0],'b',label="$\phi = 0$") #
line3 = plt.plot(k,l[1],'g',label="$\phi = \pi/16$") #
line4 = plt.plot(k,l[2],'r',label="$\phi = 2\pi/16$") #
line5 = plt.plot(k,l[3],'c',label="$\phi = 3\pi/16$") #
line6 = plt.plot(k,l[4],'m',label="$\phi = 4\pi/16$") #
#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc = 3,
#           ncol = 3, mode="expand", borderaxespad = -0.2)
plt.legend(loc = 'lower left')
plt.axis([epsilon,kmax,0.75,1.1])
plt.ylabel('$c_{FD}/c$')
plt.xlabel('$K=k\Delta x$')

#plt.savefig('/home/yajun/Documents/wave2d/figsUnused/test4-25-21remez.eps', format='eps', dpi=300)
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
