# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 15:50:05 2020

@author: Alienware
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import scipy as sp
import matplotlib.animation as animation




"""
Implementation of Smooth Particule Hydrodynamics followinf Mueller's paper.
https://matthias-research.github.io/pages/publications/sca03.pdf

Surface Tension isn't implemented, explaining unrealistic feature.

inspiration for spatial hash
https://www.gamedev.net/tutorials/_/technical/game-programming/spatial-hashing-r2697/

"""
class Hash:
    
    def __init__(self, cellsize): # taille des cellules, largeur hauteur totale
        self.cellsize = cellsize

        
        # creation cell, a dictionnary
        self.cell = {}

        """"
        dictionnary type
        key, value 
        one value per key but multiple key per value possible
        to have multiple value for one key, use list or set
        Lists are slightly faster than sets when you just want to iterate over the values.
        Sets, however, are significantly faster than lists if you want to check if an item is contained within it. 
        They can only contain unique items though
                
        best:
            dictionnary where
            key = cell
            value = set of particule (faster as we will jsut look up)
        """
        
    def coords_to_index(self, coords):    # conversion coordonnate => index
        return (coords[0] // self.cellsize, coords[1] // self.cellsize) # // : return the biggest integer dividend
    
        
    def add(self,coords,particule_id):
        """
        To add particule
        
        The setdefault(key,value) method returns the value of the item with the specified key.

        If the key does not exist, insert the key, with the specified value
        
        if not default, you should specify the type of the value, here a set
        """
        self.cell.setdefault(self.coords_to_index(coords), set()).add(particule_id)
        
    def remove(self,coords, particule_id):
        "discard dosent raise error if value is not present"
        self.cell[self.coords_to_index(coords)].discard(particule_id)
        
    def move(self, old_coords, new_coords, particule_id):
        """
        To move
            check in new_coords is in the cell
            if not => add, and remove in old_coords
            else => nothing to do
        """        
        self.remove(old_coords, particule_id)            
        self.add(new_coords, particule_id)
        
        
    def neighbours(self, coords):
        idx = self.coords_to_index(coords)        
        cell_N = None
        cell_S = None
        cell_E = None
        cell_W = None        
        cell_NE = None
        cell_NW = None
        cell_SE = None
        cell_SW = None
        if (idx[0], idx[1]+1) in self.cell: cell_N = (idx[0], idx[1]+1)
        if (idx[0], idx[1]-1) in self.cell: cell_S = (idx[0], idx[1]-1)
        if (idx[0]+1, idx[1]) in self.cell: cell_E = (idx[0]+1, idx[1])
        if (idx[0]-1, idx[1]) in self.cell: cell_W = (idx[0]-1, idx[1])
       
        if (idx[0]+1, idx[1]+1) in self.cell: cell_NE = (idx[0]+1, idx[1]+1)
        if (idx[0]-1, idx[1]+1) in self.cell: cell_NW = (idx[0]-1, idx[1]+1)
        if (idx[0]+1, idx[1]-1) in self.cell: cell_SE = (idx[0]+1, idx[1]-1)
        if (idx[0]-1, idx[1]-1) in self.cell: cell_SW = (idx[0]-1, idx[1]-1)

        return [value for cel in (idx, cell_N, cell_S, cell_E, cell_W, cell_NE, cell_NW, cell_SE, cell_SW) if cel!=None for value in self.cell.get(cel) ]
    

# set the base params
n = 1000 # nb particule
width = 400
height = 400
edge = 5
xlim = [edge, width - edge]
ylim = [edge, height - edge]

h=5

"Kernel definition"

reg_poly6 = 4/(np.pi*h**8)
reg_spiky = -30/(np.pi*h**5) # for grad kernel
reg_visc = 20/(3*np.pi*h**5) #for laplacian

xyz= []
sphash = Hash(h)

# set-up a dam-break scenario
i = 0
dam_ylim = (ylim[0], ylim[1]-1 )
dam_xlim = (int((xlim[1] - xlim[0]) * 0.0001), int((xlim[1] - xlim[0]) * 0.4))

for y in range(*dam_ylim, h):
    for x in range(*dam_xlim, h):
        if i < n:
            xyz.append( [x + np.random.uniform(0, h * 0.1), y + np.random.uniform(0, h * 0.1)])
        i += 1
        
    pass

xyz = np.asarray(xyz)

for ii in range(xyz.shape[0]):
    sphash.add(xyz[ii],ii)


n_part=xyz.shape[0]

fig, ax1 = plt.subplots()
im = ax1.scatter(xyz[:,0],xyz[:,1])

xmin,xmax = xlim[0], xlim[1] #0,2*width
ymin, ymax = ylim[0], ylim[1] #0,2*height

ax1.set_xlim([xmin-h,xmax+h])
ax1.set_ylim([ymin-h,ymax+h])

plt.show()

m = np.ones(n_part)*5

rho=np.zeros(n_part) # density
press=np.zeros(n_part) # pressure
vel=np.zeros([n_part,2]) # velocity

g = -.1 #gravity
k = 5 # constante gaz
rho0 = 1
mu = 10 # viscosity
dt = 0.01   
damp = -0.5
f_grav =  [0, g]

def update(tt,xyz, vel, rho, press):

    for ii in range(n_part):
        rho[ii] = 0
        for jj in sphash.neighbours(xyz[ii]):
            r = xyz[jj,:] - xyz[ii,:]
            r_2 = r.dot(r)

            if r_2 <= h**2:
                rho[ii] += reg_poly6 * (h**2 - r_2) ** 3

        press[ii] = k * (rho[ii] - rho0)

    f_press=np.zeros([n_part,2])
    f_visc=np.zeros([n_part,2])

    for ii in range(n_part):    #for ii in range(n_part):
        for jj in sphash.neighbours(xyz[ii]):            
            if ii==jj: continue
            
            r = xyz[jj] - xyz[ii]            
            rn = np.linalg.norm(r)
            
            if 0 < rn < h:
                f_press[ii,:] += - r/rn * m[jj]**2 * (press[ii] + press[jj]) / (2*rho[jj]) * reg_spiky * (h-rn)**2
                f_visc[ii,:] += mu * m[jj] * (vel[jj]-vel[ii]) / rho[jj] * reg_visc * (h-rn)

    "Euler integration"         
    acc = (f_press + f_visc + f_grav) / rho[:,np.newaxis]   
    vel += acc*dt   
    xyz += vel*dt
    
            
    jnk = np.argwhere((xyz[:,0]<xmin))# => u=-u
    if jnk.size !=0:
        vel[jnk,0] = damp*vel[jnk,0]
        xyz[jnk,0] = xmin

    jnk = np.argwhere((xyz[:,0]>xmax))# => u=-u
    if jnk.size !=0:
        vel[jnk,0] = damp*vel[jnk,0]
        xyz[jnk,0] = xmax
        
    jnk = np.where((xyz[:,1]<ymin) ) #=> v=-v   
    if jnk!=0:
        vel[jnk,1] = damp*vel[jnk,1]
        xyz[jnk,1] = ymin  
        
    jnk = np.where((xyz[:,1]>ymax) ) #=> v=-v   
    if jnk!=0:
        vel[jnk,1] = damp*vel[jnk,1]
        xyz[jnk,1] = ymax
        
        
    im.set_offsets(xyz)
    ax1.set_title(tt)

    "update spatial through recomputing it"           
    sphash.cell.clear()
    for ii in range(n_part):
            sphash.add(xyz[ii],ii)
                
    if tt%100 == 0:
        print(tt)

    return im,xyz,vel,rho,press

ani = animation.FuncAnimation(fig, update,5000,fargs=(xyz, vel, rho, press))

writer = animation.writers['ffmpeg'](fps=120)
ani.save('demo_sph.mp4',writer=writer)