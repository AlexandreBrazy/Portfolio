# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:11:57 2020

@author: Alienware
"""
"""
https://arxiv.org/ftp/arxiv/papers/1510/1510.08224.pdf
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6266048/
https://github.com/rlee32/lbm_matlab/blob/master/navier_stokes/vc/collide_mrt_vcs.m
https://pdf.sciencedirectassets.com/277910/1-s2.0-S1876610212X00043/1-s2.0-S1876610212001130/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjED8aCXVzLWVhc3QtMSJGMEQCIBczHWdyDXkzP0mvlX0jTyTJgGqyrlyl9Z%2BUOf5ynShCAiBbUQvOaupKU7V%2BvQjNPDhY925ObSNhCAZhjTNQnDeTnyq9AwjI%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAMaDDA1OTAwMzU0Njg2NSIM8VLqxACqv9oJxkHKKpEDLQJNCahZIXGXXfjPVs0bxW585Jo5CRsZjKKvl5xcfjeXQsR3RVvzG%2F5bj6arn1%2B8BgV2dRY352fRlxJk7uw9y%2BAjqD46E63OxxvP%2FB8Vv8SJ5nv3W9o9uDsN4Gak0Vt4QIJzyzYv5ACFrqq9fmASlVMLwCJM7T2RnkbN4bExfgyCS74RKtH29n5pbRVpeenADx42f8olrtkGieisBZzVBl1PpCIJ96MWjkNco52ekHHlisabExSq292XJ%2B5uGt2xCobKd0TI9fZFQB5faLI2yEfK8zUvbv0qYVpGVYo6HrHm7iZsyPC5%2FEazoH4hnKhss%2BZw7P6g%2FgCcR3OBOnETvvTcCVAJEp44jTp1v5iJ107G7DFOkUOP40kCR99y9cgZw6ZUyl47KMpLLgK0XnyWlkej53C7rKHuQ83H7oUSGjglH6N%2Fc4XHqmqz60FcPTysu5DgIPQpnfjPc0fPjcnTQR9ifcvOEsZr7tKuxsyowNcv2Hg3QNlkwm0R9%2BeS2SqCOWvCmCEKL%2FGdtY9%2Fc%2FrAwj4wpMyl%2FgU67AHZsPB8IL%2BY9puhw%2FkqiU8OitsSkmPV5LmJ2VRXSS8v0NbSBuINe%2BVllTM%2BtDYGGFXpvoMaQkSTigvqbnnMnLLZlvq6LkZHwRzfkizd4NC3ZNrYCRP%2B%2FrsYQYPcXxofh24z9A4EpO%2BgZdfJWPq%2BKj%2FdShM5sHD9qgk%2FdUGYYmJXd61qJ22lmip37R5mCg0SafhWhekNuyZnWZ5HMrNjZJalAys9MHxgN7tMRmV4Hk%2Bccq0pvbH7H3D7r6QzRJi6G6pQ3g2wGHIZuVsI1mGohH2%2FZwP%2B9ojMqLRPcy9b4aioHg32wlAw2Gq0E8OnFw%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20201203T232854Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY4ZQNXHVU%2F20201203%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=ec5d5485e2db9ca0f048ac3cd08f6edaeef14bf8078c189783bd8ee4c589888a&hash=5fafd53d2e7e6ad4832ac14512ddecc80303516fff27dfd48f796c275e4c843f&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1876610212001130&tid=spdf-06fe1c7f-c378-464c-8f01-907ccb9ea482&sid=1871e86f178aa445d4-b2e2-c5f3c1ee5346gxrqa&type=client
viscosity counteraction
https://doc.global-sci.org/uploads/Issue/AAMM/v1n8/18_37.pdf
surtout:
    http://www.ijmmm.org/papers/149-TT3003.pdf

"""

import numpy as np
import matplotlib.pyplot as plt
import pylab
import matplotlib.animation as animation


def curl(): #Functions related to color when plotting
    return  np.roll(uy,-1,axis=1) - np.roll(uy,1,axis=1) - np.roll(ux,-1,axis=0) + np.roll(ux,1,axis=0)

def stream():
    global f, ex, ey
    
    for ii in range(1,9):
        f[:,:,ii]=np.roll(np.roll(f[:,:,ii],ex[ii],axis=1),ey[ii], axis=0) 

def collide():
    global f, m, M, S, MinvS, Minv, rho, ux, uy
    
    m=np.tensordot(M,f,axes=((1),(2))) # generalization of M*f[i,j,:], axes=... let you choose the axes on wich you multiply
    
    """
    m=(rho,r or e depending on source, epsilon,jx,qx,jy,qy,pxx, pyy)    
    jx, jy = rho*ux, rho*uy
    """  
    rho  = m[0,:,:]
    ux = m[3,:,:]/rho
    uy = m[5,:,:]/rho
    
    
    jx = m[3,:,:]
    jy = m[5,:,:]

       
    meq = np.array([rho, 
                -2*rho+3*(jx**2+jy**2)/rho,
                rho-3*(jx**2+jy**2)/rho,
                jx,
                -jx,
                jy,
                -jy,
                (jx**2-jy**2)/rho,
                jx*jy/rho])
#    
    
    f = f - np.moveaxis(np.tensordot(MinvS,m-meq, axes=((1),(0))), 0, -1 ) # np.moveaxis because im too lazy too correct the whole code for array index and i want to avoid loop


def BB(): # no slip
    global f, BC
    for ii in [1,2,5,6]:
            jnk=f[BC,ii+2]
            f[BC,ii+2]=f[BC,ii]
            f[BC,ii]=jnk
            

BC = pylab.imread('rod.png')[:,:,0]
BC = np.array(BC, dtype=bool)
BC=np.invert(BC) # bc blanc = true

BC[0,:]=True
BC[-1,:]=True

Npt = 100 # point per real unit (cm)
Lx = 1
Ly = 1

"dimension"
Nx=BC.shape[1]
Ny=BC.shape[0]

#Nx = Lx * Npt
#Ny = Ly * Npt

"direction"
ex=np.array([0,1,0,-1,0,1,-1,-1,1])
ey=np.array([0,0,1,0,-1,1,1,-1,-1])

"weight"
w=np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36]) # weigth

dx = 1/Npt
dt = 0.01 # decrease to increase stability
 
c=dx/dt

rho = 1
ux0=0.1

#Re=ux*Lx/nu
nu=5e-5
Re=ux0*Lx/nu
#nu = ux0*(Nx/Npt)/Re # viscosite
cs = c/np.sqrt(3)
omega = 1/(nu/(cs**2*dt)+0.5) # nu = dt*(1/ omega -1/2)cs**2

print(Re)

M = np.array([
        [1,1,1,1,1,1,1,1,1],
        [-4,-1,-1,-1,-1,2,2,2,2],
        [4,-2,-2,-2,-2,1,1,1,1],
        [0,1,0,-1,0,1,-1,-1,1],
        [0,-2,0,2,0,1,-1,-1,1],
        [0,0,1,0,-1,1,1,-1,-1],
        [0,0,-2,0,2,1,1,-1,-1],
        [0,1,-1,1,-1,0,0,0,0],
        [0,0,0,0,0,1,-1,1,-1]
        ])
    
s0 = 1
s1 = 1.4
s2 = 1.4
s3 = 1
s4 = 1.2
s5 = 1
s6 = 1.2
s7 = omega
s8 = omega

S = np.diag([s0,s1,s2,s3,s4,s5,s6,s7,s8])

Minv = np.linalg.inv(M)
MinvS = Minv@S

"set init condition"

f = np.ones([Ny,Nx,9])
#
# To enforce initial condition with motion
#
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

m=np.tensordot(M,f,axes=((1),(2))) # generalization of M*f[i,j,:]




def update(i):
        stream()
        BB()
#		 Impose right motion on the left border
#        f[:,0,1] = 1/9 * (1 + 3*ux0 + 4.5*ux0**2 - 1.5*ux0**2)
#        f[:,0,3] = 1/9 * (1 - 3*ux0 + 4.5*ux0**2 - 1.5*ux0**2)
#        f[:,0,5] = 1/36 * (1 + 3*ux0 + 4.5*ux0**2 - 1.5*ux0**2)
#        f[:,0,8] = 1/36 * (1 + 3*ux0 + 4.5*ux0**2 - 1.5*ux0**2)
#        f[:,0,6] = 1/36 * (1 - 3*ux0 + 4.5*ux0**2 - 1.5*ux0**2)
#        f[:,0,7] = 1/36 * (1 - 3*ux0 + 4.5*ux0**2 - 1.5*ux0**2)
        collide()
        im.set_array(curl()- BC[:, :])#,cmap="jet",norm=plt.Normalize(-.1, .1))
        im2.set_array(np.sqrt(uy**2+ux**2)) #curl() - BC[:, :, 0])
        print(i)
        return im

#fig = plt.figure()

fig, ax = plt.subplots(2)
#im[0].plot(un)


stream()
BB()
collide()
im = ax[0].imshow(curl(), cmap='jet',norm=plt.Normalize(-.1, .1)) # - barrier[:, :, 0],cmap="jet")
im2=ax[1].imshow(np.sqrt(uy**2+ux**2), cmap='jet',norm=plt.Normalize(-.5, .5)) # - barrier[:, :, 0],cmap="jet")
#im2 = ax[1].imshow(np.sqrt(uy**2+ux**2),cmap='jet')
ani = animation.FuncAnimation(fig, update,60000)
###ani = animation.FuncAnimation(fig,anim,60000)
#writer = animation.writers['ffmpeg'](fps=100)
###
#ani.save('demo_lbm.mp4',writer=writer)

#plt.colorbar()
plt.show()