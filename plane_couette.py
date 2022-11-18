#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 18:19:23 2021

@author: pietro
"""

import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

def couette(t,uq_init):
    #Assign vectors
    q1=uq_init[0:N]
    q2=uq_init[N:2*N]
    u1=uq_init[2*N:3*N]
    u2=uq_init[3*N:4*N]
    
    #Define second and first derivative with respect to x
    dxxq1= (np.roll(q1,-1)- 2*q2 + np.roll(q1,1))/dx**2 #np.roll translates the elements in the array to the left -1 or right +1
    dxxq2= (np.roll(q2,-1)- 2*q2 + np.roll(q2,1))/dx**2
    dxq1= (np.roll(q1,-1)- np.roll(q1,1))/(2*dx)
    dxq2= (np.roll(q2,-1)- np.roll(q2,1))/(2*dx)
    dxu1= (np.roll(u1,-1)- np.roll(u1,1))/(2*dx)
    dxu2= (np.roll(u2,-1)- np.roll(u2,1))/(2*dx)
    
    print(t)   #check the current evaluation time for debugging
    
    #Dynamical system equations of plane Couette
    dq1 = q1*(u1+r-1-(r+d)*(q1-1)**2)+dxxq1 +k*(q2-q1) -U*dxq1
    du1 = e1*(1-u1) -e2*u1*q1 -dxu1 -U*dxu1
    dq2 = q2*(u2+r-1-(r+d)*(q2-1)**2)+dxxq2 +k*(q1-q2) +U*dxq2
    du2 = e2*(1-u2) -e2*u2*q2 +dxu2 +U*dxu2
    
    #Concatenate results into 1D vector
    duq12=np.concatenate((dq1,dq2,du1,du2))
    return [duq12]

L=200 #size of domain in x-direction
U=1
N = 1000 #number of discretization in x-direction
x= np.linspace(-L/2,L/2,N)
dx = x[1]-x[0] 
T = 20 #time of solution

#Assign parameters
r=0.7
d=0.1
k=0.1
e1=0.1
e2=0.1

#Initial conditions
q1=np.exp(-x**2/10)
q2 = np.full(N,0)
u1 = np.full(N,0)
u2 = np.full(N,0)

#Concatenate initial conditions to 1D vector for solve_ivp
uq_init=np.concatenate((q1,q2,u1,u2))

#Solve dynamical system through Runge-Kutta in time and finite differences in x
sol = spi.solve_ivp(couette,(0,T),uq_init,method='RK45')
q1_sol=sol.y[0:N,:]
q2_sol=sol.y[N:2*N,:]
u1_sol=sol.y[2*N:3*N,:]
u2_sol=sol.y[3*N:4*N,:]
t=sol.t

#Reduce dimension of vectors for plotting
n=5 #take every nth result in time
q1_solp=q1_sol[:,::n]
q2_solp=q2_sol[:,::n]
u1_solp=u1_sol[:,::n]
u2_solp=u2_sol[:,::n]
tt=t[::n]

X, Y = np.meshgrid(x, tt)

plt.figure(0)
plt.contourf(X, Y, u1_solp.T)
plt.title('Evolution of u1');
plt.xlabel("x")
plt.ylabel("t")

plt.figure(1)
plt.contourf(X, Y, u2_solp.T)
plt.title('Evolution of u2');
plt.xlabel("x")
plt.ylabel("t")

plt.figure(2)
plt.contourf(X, Y, q1_solp.T)
plt.title('Evolution of q1');
plt.xlabel("x")
plt.ylabel("t")

plt.figure(3)
plt.contourf(X, Y, q2_solp.T)
plt.title('Evolution of q2');
plt.xlabel("x")
plt.ylabel("t")

plt.show()

#plotting
# X, Y = np.meshgrid(x, tt)

# plt.figure(0)
# ax1 = plt.axes(projection='3d')
# ax1.plot_surface(X, Y, u1_solp.T, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
# ax1.set_title('Evolution of u1');
# ax1.set_xlabel("x")
# ax1.set_ylabel("t")
# ax1.set_zlabel("u1")

# plt.figure(1)
# ax2 = plt.axes(projection='3d')
# ax2.plot_surface(X, Y, u2_solp.T, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
# ax2.set_title('Evolution of u2');
# ax2.set_xlabel("x")
# ax2.set_ylabel("t")
# ax2.set_zlabel("u2")

# plt.figure(2)
# ax3 = plt.axes(projection='3d')
# ax3.plot_surface(X, Y, q1_solp.T, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
# ax3.set_title('Evolution of q1');
# ax3.set_xlabel("x")
# ax3.set_ylabel("t")
# ax3.set_zlabel("q1")

# plt.figure(3)
# ax4 = plt.axes(projection='3d')
# ax4.plot_surface(X, Y, q2_solp.T, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
# ax4.set_title('Evolution of q2');
# ax3.set_xlabel("x")
# ax3.set_ylabel("t")
# ax4.set_zlabel("q2")

# plt.show()
