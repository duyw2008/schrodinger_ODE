# -*- coding: utf-8 -*-
'''
求解薛定谔方程
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# 注意zeros([10,1])和zeros([10,])是有区别的，这里应该用zeros([10,])，其余同
##############################################################################
#----- Numerical integration of ODE via fixed-step classical Runge-Kutta -----
def RK4Step(odefunc, t,u,h):
    k1 = odefunc(t,u)
    k2 = odefunc(t+0.5*h, u+0.5*k1*h)
    k3 = odefunc(t+0.5*h, u+0.5*k2*h)
    k4 = odefunc(t+h,     u+k3*h)
    return u + (k1+2*k2+2*k3+k4)*(h/6.)

################################################################################
def odefunc(t,u):
    '''
    u_t=f(x,u)=i*((u(x+dx,t)+u(x-dx,t)-2*u(x,t))/(2*dx**2)-V*u)
    '''
    u_left =np.array(np.zeros([numx,]),dtype=complex)
    u_right=np.array(np.zeros([numx,]),dtype=complex)

    u_left[0:numx-2,]=u[1:numx-1,]
    u_left[numx-1,]  =u_left[numx-2,]
    u_right[1:numx,] =u[0:numx-1,]
    u_right[0,]      =u_right[1,]

    u_t = 0.5j*hbar/m*(u_left + u_right - 2*u)/(dx**2)-1j*V_x*u
    return u_t

################################################################################
# Utility functions for running the animation
def square_barrier(x, width, height):
    Vx = np.zeros(x.shape)
    Vx[x >0] = height
    Vx[x >0+width] = 0.0
    return Vx

################################################################################
# Helper functions for gaussian wave-packets
def gauss_x(x, x0):
    '''
    a gaussian wave packet of width mu, centered at x0, with momentum k0
    '''
    mu=1.0
    ka=60
    sigma=0.1
    return mu*np.exp(ka*1j*x)*np.exp(-(x-x0)**2/(2*sigma**2))

################################################################################
hbar=1.
m   =1.
t0  =0.0
dx  =0.01
dt  =0.0001
numx=2**11
V0  =2.0
L   =numx*dx/2.0
x   =dx * (np.arange(numx) - 0.5 * numx)

# 计算频率取值范围及步长
dk = 2 * np.pi / (numx * dx)
k0 =-0.5 * numx * dk
k  = k0 + dk * np.arange(numx)

# Potential
V_x  = square_barrier(x, 20*dx, V0*0)
V_x[x<-0.99*L] = 2E2
V_x[x> 0.99*L] = 2E2
   
# Gauss Package
x0 =-200*dx
u0 = gauss_x(x, x0)    # 高斯波包作为u函数在x的初值
u  = u0
t  = t0
################################################################################
# ODE method
for ii in range(300): 
    u = RK4Step(odefunc,t,u,dt)
    t=t+dt
uhat = np.fft.fft(u)

plt.figure()
ax1 = plt.subplot(211,xlim=(-L-2,L+2),ylim=(-0.2,V0*1.2))
ax1.plot(x,np.real(np.abs(u)),x,np.real(np.abs(u))**2,x,V_x)
ax1.set_ylim(0,1.2*V0)

ax2 = plt.subplot(212,xlim=(k[0,],k[numx-1,]),ylim=(-1,250))
ax2.plot(k,uhat)
plt.show()
################################################################################
# 差分法
'''
B0 = gauss_x(x, x0)    # 高斯波包作为u函数在x的初值
B=np.array(np.zeros([numx,numt]),dtype=complex)
C=np.array(np.zeros([numx,numt]),dtype=complex)
   
B[:,0]=np.transpose(B0)
A=np.diag(-2+2j+V_x)+np.diag(np.ones([numx-1,]),1)+np.diag(np.ones([numx-1,]),-1)

for t in range(0,numt-1,1):
    C[:,t+1]=4j*np.linalg.solve(A,B[:,t])
    B[:,t+1]=C[:,t+1]-B[:,t]

nn=60
plt.figure()
plt.grid()
plt.plot(x,np.abs(B[:,nn])**2,x,V_x)
plt.ylim(0,V0*1.2)
plt.show()
'''
