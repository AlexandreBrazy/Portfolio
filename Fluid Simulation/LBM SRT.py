# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 14:27:23 2020

@author: Alienware
"""
"""
source:
    https://personal.ems.psu.edu/~fkd/courses/EGEE520/2017Deliverables/LBM_2017.pdf
    https://www.mathworks.com/matlabcentral/fileexchange/48103-the-lattice-boltzmann-method-in-25-lines-of-matlab-code-lbm
    https://www.youtube.com/watch?v=Y2K4EEYz27w
    
Lattice D2Q9:

  6_       2        5
  |\       |       /
      \    |   /
  3 <----- 0 -----> 1
         / |    \
     /     |        \
  7        4        8
  
  algo:
      1 initialise f and macro values
      ie f, w, ex, ey...
      2 solve equilibrium
      3 collision
      4 streaming
      5 update macro value
      6 repeat from 2
     
The code compute the flow of a fluid using the lattice boltzmann method with single relaxation time.
the obstacle are imported through a black and white png file
the result is save as a video file

The aim of this code is to help the comprehension of the LBM method. 
	 
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import pylab


def curl(): #Functions related to color when plotting
    return  np.roll(uy,-1,axis=1) - np.roll(uy,1,axis=1) - np.roll(ux,-1,axis=0) + np.roll(ux,1,axis=0)

def stream():
    global f, ex, ey
    
    for ii in range(1,9):
        f[:,:,ii]=np.roll(np.roll(f[:,:,ii],ex[ii],axis=1),ey[ii], axis=0) 
        
def collide():
    global f, w, ex, ey, rho, ux, uy
    
    rho= np.sum(f, axis=2)
    ux=np.sum(f*ex, axis=2)/rho # f*ex multiplie par le sens de propagation 
#    ux=ux+deltaux
    uy=np.sum(f*ey, axis=2)/rho
    
    u_sqr=ux**2+uy**2
    
    for ii in range(9):
            eu= ux*ex[ii]+uy*ey[ii]
            f[:,:,ii]= (1-omega)*f[:,:,ii]+omega*w[ii]*rho*(1 + 3*eu/c**2 + 9/2 *(eu/c**2)**2 - 3/2 *u_sqr/c**2)

def BB(): # no slip
    global f, BC
    for ii in [1,2,5,6]:
            jnk=f[BC,ii+2]
            f[BC,ii+2]=f[BC,ii]
            f[BC,ii]=jnk
			
			
"""read a Black and white png image to set the obstacle. Black is the obstacle"""
BC = pylab.imread('rod.png')[:,:,0] 
BC = np.array(BC, dtype=bool)
BC=np.invert(BC) # because white = true

"dimension"
Nx=BC.shape[1]
Ny=BC.shape[0]

ux0=0.1

"time/space step"
dt=1
dx=1
dy=1

"direction"
ex=np.array([0,1,0,-1,0,1,-1,-1,1])
ey=np.array([0,0,1,0,-1,1,1,-1,-1])

"relaxation parameter formulation is absolutly not correct..."
c=dx/dt
rho0=1
Re=300 # Reynolds number rho*V*L/mu
mu=1/Re #Re=u*L/nu => nu=u*L/Re https://palabos-forum.unige.ch/t/lattice-boltzmann-units/28 nu_{Phys}=dt/(dx*dx)nu_{LB}=dt/(dxdx)*c_s^2(tau-1/2). ui/cs<<1
omega= 1/(3*mu+0.5)


"weight"
w=np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36]) # weigth

rho=9

f=np.ones([Ny,Nx,9])/rho
#
#BC = np.zeros((Ny, Nx), bool)
#BC[:,0]=True
#BC[:,-1]=True
#up and bottom wall are obstacle
BC[0,:]=True
BC[-1,:]=True

"set speed in the whole fluid"
#f[:,:,0] = w[0] * (np.ones((Ny,Nx)) - 1.5*ux0**2)	# particle densities along 9 directions
#
#f[:,:,1] = w[1] * (np.ones((Ny,Nx)) - 1.5*ux0**2)
#f[:,:,2] = w[2] * (np.ones((Ny,Nx)) - 1.5*ux0**2)
#f[:,:,3] = w[3] * (np.ones((Ny,Nx)) + 3*ux0 + 4.5*ux0**2 - 1.5*ux0**2)
#f[:,:,4] = w[4] * (np.ones((Ny,Nx)) - 3*ux0 + 4.5*ux0**2 - 1.5*ux0**2)
#
#f[:,:,5] = w[5] * (np.ones((Ny,Nx)) + 3*ux0 + 4.5*ux0**2 - 1.5*ux0**2)
#f[:,:,6] = w[6] * (np.ones((Ny,Nx)) + 3*ux0 + 4.5*ux0**2 - 1.5*ux0**2)
#f[:,:,7] = w[7] * (np.ones((Ny,Nx)) - 3*ux0 + 4.5*ux0**2 - 1.5*ux0**2)
#f[:,:,8] = w[8] * (np.ones((Ny,Nx)) - 3*ux0 + 4.5*ux0**2 - 1.5*ux0**2)

"set speed just on the edge"
f[:,0,1] = 1/9 * (1 + 3*ux0 + 4.5*ux0**2 - 1.5*ux0**2)
f[:,0,3] = 1/9 * (1 - 3*ux0 + 4.5*ux0**2 - 1.5*ux0**2)
f[:,0,5] = 1/36 * (1 + 3*ux0 + 4.5*ux0**2 - 1.5*ux0**2)
f[:,0,8] = 1/36 * (1 + 3*ux0 + 4.5*ux0**2 - 1.5*ux0**2)
f[:,0,6] = 1/36 * (1 - 3*ux0 + 4.5*ux0**2 - 1.5*ux0**2)
f[:,0,7] = 1/36 * (1 - 3*ux0 + 4.5*ux0**2 - 1.5*ux0**2)

def update(i):
        stream()
        BB()
        f[:,0,1] = 1/9 * (1 + 3*ux0 + 4.5*ux0**2 - 1.5*ux0**2)
        f[:,0,3] = 1/9 * (1 - 3*ux0 + 4.5*ux0**2 - 1.5*ux0**2)
        f[:,0,5] = 1/36 * (1 + 3*ux0 + 4.5*ux0**2 - 1.5*ux0**2)
        f[:,0,8] = 1/36 * (1 + 3*ux0 + 4.5*ux0**2 - 1.5*ux0**2)
        f[:,0,6] = 1/36 * (1 - 3*ux0 + 4.5*ux0**2 - 1.5*ux0**2)
        f[:,0,7] = 1/36 * (1 - 3*ux0 + 4.5*ux0**2 - 1.5*ux0**2)
        collide()
        im.set_array(curl()- BC[:, :])
        im2.set_array(np.sqrt(uy**2+ux**2)) 
        return im


fig, ax = plt.subplots(2)

stream()
collide()
im = ax[0].imshow(curl(), cmap='jet',norm=plt.Normalize(-.1, .1)) # show the curl
im2=ax[1].imshow(np.sqrt(uy**2+ux**2), cmap='jet') # show the speed
ani = animation.FuncAnimation(fig, update,60000)
writer = animation.writers['ffmpeg'](fps=100)
##
ani.save('demo_lbm.mp4',writer=writer)

plt.colorbar()
plt.show()