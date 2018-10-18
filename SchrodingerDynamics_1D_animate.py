# -*- coding: utf-8 -*-
'''

'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# 注意zeros([10,1])和zeros([10,])是有区别的，这里应该用zeros([10,])，其余同
##############################################################################
#----- Numerical integration of ODE via fixed-step classical Runge-Kutta -----

def RK4Stream(odefunc,u_init,t_init,h):
    u = u_init
    t = t_init
    i=1
    while True:
        u = RK4Step(odefunc, t, u, h)
        t = t+h
        yield t,u

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
    u_left=np.array(np.zeros([numx,]),dtype=complex)
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
    mu=1.
    ka=2.
    sigma=15.
    return mu * np.exp(ka*1j*x)*np.exp(-(x-x0)**2/(2*sigma**2))

################################################################################
hbar=1.
m   =1.
t0  =0.0
dx  =0.2
dt  =0.03
numx=2**11
V0  =2.0
L   =numx*dx/2.0
x   =dx * (np.arange(numx) - 0.5 * numx)

# 计算频率取值范围及步长
dk = 2 * np.pi / (numx * dx)
k0 =-0.5 * numx * dk
k  = k0 + dk * np.arange(numx)

# Potential
V_x  = square_barrier(x, 20*dx, V0)
V_x[x<-0.98*L] = 2E1
V_x[x> 0.98*L] = 2E1
   
# Gauss Package
x0=-300*dx
u0 = gauss_x(x, x0)    # 高斯波包作为u函数在x的初值

################################################################################
# ODE method
fig = plt.figure()

#'top axes show the x-space data'
ax1      = plt.subplot(211,xlim=(-L-2,L+2),ylim=(-0.2,V0*1.2))
line11,  = ax1.plot([],[], c='r')
line12,  = ax1.plot([],[], c='b')
V_x_line,= ax1.plot([],[], c='k', label=r'$V(x)$')
V_x_line.set_data(x,V_x)
ax1.set_xlabel('$x$')
ax1.set_ylabel(r'$\psi(x)$')

#'bottom axes show the k-space data' 
ax2     = plt.subplot(212,xlim=(k[0,],k[numx-1,]),ylim=(-1,250))
line2,  = ax2.plot([],[])
ax2.set_xlabel('$k$')
ax2.set_ylabel(r'$\psi(k)$')

u_stream = RK4Stream(odefunc,u0,t0,dt)
def animate(i):
    t,u = next(u_stream)
    uhat = np.fft.fft(u)
    line11.set_data(x,np.real(np.abs(u)))     # 波函数（概率幅）
    #line12.set_data(x,np.real(np.abs(u))**2) # 概率    
    line2.set_data(k,np.real(np.abs(uhat)))   # 频率
    return line11,line12,line2

anim = animation.FuncAnimation(fig, animate,frames=30, interval=1, blit=True)
#ax1.grid()
#ax2.grid()
plt.show()

