#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 18:13:36 2020

This code modifies wave2d_x250.py and wave2d_may10.py. Wave propagation in 2D square. With initial condition Ricker wavelet. Ricker wavelet is the source term.

Improvements/concerns:
    1. saves figures as .eps files at preset time steps
    2. introduce parameters, especially be careful with gamma to be consistent
    3. checker board pattern indicates instability
    4. initial conditions are Ricker wavelet at time 0 and dt

@author: yajun
"""
import numpy as np
import matplotlib.pyplot as plt
import os, os.path
############## define parameters
figsavepath = os.path.abspath('/home/yajun/Documents/wave2d/figsPropGrid200')

config = dict()
	
	# Configure the wavelet
config['nu0'] = 10 #Hz
	
	# Physical domain parameters
config['x_limits'] = [0.0, 6.0] 
config['y_limits'] = [0.0, 6.0] 
config['nx'] = 201 #101 
config['dx'] = (config['x_limits'][1] - config['x_limits'][0]) / (config['nx']-1)
config['ny'] = 201 #101 
config['dy'] = (config['y_limits'][1] - config['y_limits'][0]) / (config['ny']-1)
	# Source position
config['x_s'] = 3.0
config['y_s'] = 3.0
	
	# wave speed
C0 = 1
	
	# Set CFL constant default value = 1.0/6.0
config['gamma'] = 1.0 / 6.0 # 0.3 # 
	
	# Define time step parameters
config['T'] = 5
config['dt'] = config['gamma'] * config['dx'] / C0
config['nt'] = int(config['T'] / config['dt'])


    # visualization parameter
step = [270,570] #[150, 220] #  [75, 135] #    
################### coefficient code details and results

c=np.c = np.array([[-3.06767651e-01],
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


config['M'] = np.size(c)-1
############### Ricker wavelet
def ricker(t, config):
	
    nu0 = config['nu0']
	
    sigma = 1./(np.pi * nu0 * np.sqrt(2))
	
    t0 = 6*sigma # guarantee causality
	
    x = (np.pi * nu0 * (t-t0))**2
	
    return (1-2*x)*np.exp(-x)

############## Point Source
t = 0
value = ricker(t,config)

location = [config['x_s'],config['y_s']]

def point_source(value, location, config):
	
    nx = config['nx']
    dx = config['dx']
    ny = config['ny']
    dy = config['dy']
     
	
    x_s,y_s = location
	
    idx = int(np.ceil(x_s / dx))
    idy = int(np.ceil(y_s / dy))
	
    delta_x = np.zeros((nx,1))
    delta_y = np.zeros((ny,1))
    delta_x[idx] = 1./dx
    delta_y[idy] = 1./dy
    result = value*delta_x*delta_y.T 
    
    return result

############ initial conditions
num = config['nx'] 
time = config['nt']
dt = config['dt']
# u index in time from 0 to time-1
u = np.zeros((num,num,time))
u[:,:,0] =  point_source(ricker(0,config),location,config) 
# np.zeros((num,num))#
u[:,:,1] =  point_source(ricker(dt,config),location,config) 
# np.zeros((num,num))#

M = config['M']
for n in range(2,time-1):
    
    he = np.zeros((num,num))
    for m in range(0,M+1):
        he = he + c[m]*(np.roll(u[:,:,n],m,axis = 0) + np.roll(u[:,:,n],-m,axis = 0) + \
        np.roll(u[:,:,n],m,axis = 1) + np.roll(u[:,:,n],-m,axis = 1))
        
    u[:,:,n+1] = - he - u[:,:,n-1] + \
    point_source(ricker(n*dt,config),location,config) 

####### visualize
fig1 = plt.figure()

 
    
plt.imshow(u[:,:,step[0]],extent=[config['x_limits'][0],config['x_limits'][1],config['y_limits'][0],config['y_limits'][1]])
    
plt.title('Wave propagation at ' + str(step[0]) + '-th time step')
fig1.savefig(figsavepath + '/' + 'remez2dgroup_025pi03gamma2l10m' + str(step[0]) + 'step.eps', dpi = 300)


####### visualize
fig2 = plt.figure()

 
    
plt.imshow(u[:,:,step[1]],extent=[config['x_limits'][0],config['x_limits'][1],config['y_limits'][0],config['y_limits'][1]])
    
plt.title('Wave propagation at ' + str(step[1]) + '-th time step')
fig2.savefig(figsavepath + '/' + 'remez2dgroup_025pi03gamma2l10m' + str(step[1]) + 'step.eps', dpi = 300)
    
    
plt.show() 

















