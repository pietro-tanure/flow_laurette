#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 18:19:23 2021

@author: pietro
"""

import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

def heat_eq(t,h):
    print(t)   # to check the current time for debugging
    hxx = (np.concatenate((np.roll(h,-1)[:-1],[0])) - 2*h + np.concatenate(([0],np.roll(h,1)[1:])))/0.02020202**2 #np.roll
    
    #Heat equation
    dhdt = hxx
    return dhdt

N = 100 #number of discretization in x-direction
x = np.linspace(-1,1,N) 
dx = x[1]-x[0] 
T = 50e-2 #time of solution

#Initial condition
h0 = ((1+x)*[x<=0]+(1-x)*[x>0]).reshape(100)

#Solve dynamical system through Runge-Kutta in time and finite differences in x
sol = spi.solve_ivp(heat_eq,(0,T),h0,method='RK45')
z=sol.y
t=sol.t

#Reduce dimension of vectors for plotting
n=4 #take every nth result in time
tt=t[::4]
zz=z[:,::4]

#Plotting
X, Y = np.meshgrid(tt, x)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, zz, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_title('surface');


