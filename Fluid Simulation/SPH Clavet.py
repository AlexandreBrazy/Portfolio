# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 19:04:15 2021

@author: Alexandre Brazy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import itertools
import numba as nb
# import tensorflow as tf

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
        "better version than the one use in mueller"
        idx = self.coords_to_index(coords) 
        "dict.get(key, default) check if value exist a return default if it doesnt"

        cell_N = self.cell.get((idx[0], idx[1]+1),None)
        cell_S = self.cell.get((idx[0], idx[1]-1),None)
        cell_E = self.cell.get((idx[0]+1, idx[1]),None)
        cell_W = self.cell.get((idx[0]-1, idx[1]),None)
        
        cell_NE = self.cell.get((idx[0]+1, idx[1]+1),None)
        cell_NW = self.cell.get((idx[0]-1, idx[1]+1),None)
        cell_SE = self.cell.get((idx[0]+1, idx[1]-1),None)
        cell_SW = self.cell.get((idx[0]-1, idx[1]-1),None)

        "return a list of set, if set not equal to default (None)"
        "self.cell.get(idx) bc we initailly use idx as a key for dict search"
        temp = [cel for cel in (self.cell.get(idx), cell_N, cell_S, cell_E, cell_W, cell_NE, cell_NW, cell_SE, cell_SW) if cel!=None]
        
        "https://stackoverflow.com/questions/30773911/union-of-multiple-sets-in-python"
        "https://www.geeksforgeeks.org/python-convert-set-into-a-list/"
        return list(itertools.chain.from_iterable(temp))
    
    

def double_density_relaxation(xyz, rho0, dt, h, k, k_near, sphash, borderhash,border_particle):
    
    
    # rho, rho_near, pressure pressure_near, k_near not use anywhere else
    rho = np.zeros(xyz.shape[0])
    rho_near = np.zeros_like(rho)
    pressure = np.zeros_like(rho)
    pressure_near = np.zeros_like(rho)  
    
    jnk=np.copy(xyz)

    
    for ii in range(xyz.shape[0]):
        jnx = sphash.neighbours(jnk[ii])
        bjnx = []#borderhash.neighbours(jnk[ii])
        for jj in jnx:
            if ii==jj : continue    
            rij = xyz[jj] - xyz[ii]
            rij_norm = np.linalg.norm(rij) 
            q = rij_norm / h

            if q < 1:
                rho[ii] += (1-q)**2
                rho_near[ii] += (1-q)**3
                
        if  bjnx: # empty list are false by default, strange but thats the recommended test
            for jj in bjnx:
                # dont need to check if ii==jj as the two represent different set of particles
                rij = border_particle[jj] - xyz[ii]
                rij_norm = np.linalg.norm(rij)  
                q = rij_norm / h
    
                if q < 1:
                    rho[ii] += (1-q)**2
                    rho_near[ii] += (1-q)**3
    
        pressure[ii] = k*(rho[ii] - rho0)
        pressure_near[ii] = k_near * rho_near[ii]
    
        dx = np.zeros(xyz.shape[1])
    
    
        for jj in jnx: 
            if ii==jj : continue    
            rij = xyz[jj] - xyz[ii]
            rij_norm =  np.linalg.norm(rij) 
            q = rij_norm / h
            
            if q < 1: 
                D = dt**2*( pressure[ii]*(1-q) + pressure_near[ii]*(1-q)**2 ) * rij/rij_norm
                xyz[jj] += D/2
                dx -= D/2          
                
 
        # if bjnx :
        #     for jj in jnx: 
        #         # dont need to check if ii==jj as the two represent different set of particles
        #         rij = border_particle[jj] - xyz[ii]
        #         rij_norm =  np.linalg.norm(rij)
        #         q = rij_norm / h
                
        #         if q < 1: 
        #             D = dt**2*( pressure[ii]*(1-q) + pressure_near[ii]*(1-q)**2 ) * rij/rij_norm
        #             dx -= D/2          

        xyz[ii] += dx
        
    return xyz

# to be add    
# def spring_displacements():
#     dt, k_spring, h
#     for spring in range():
#         rij = xyz[jj] - xyz[ii]
#         rij_norm = np.linalg.norm(rij)
        
#         D = dt**2 * k_spring*(1 - Lij/h)*(Lij-rij_norm)*rij
#         xyz[ii] -= D/2
#         xyz[jj] += D/2
        
# def spring_adjustment():
    
def viscosity_impulse(xyz, vel, sigma, beta, dt, h, sphash):
    for ii in range(xyz.shape[0]):
        jnx = sphash.neighbours(xyz[ii])
        
        for jj in jnx: #range(xyz.shape[0]):
            if ii==jj : continue    
            rij = xyz[jj] - xyz[ii]
            rij_norm =  np.linalg.norm(rij)#rij /
            q = rij_norm / h
                        
            if q < 1: 
                u = (vel[ii] - vel[jj]) @ (rij/(rij_norm)) # @= dot product ; classic softening to go around particle in corner
                
                if u>0:
                    'sigma ++ => ++ viscous'
                    'sigma  == 0 for less viscous'
                    I = dt*(1-q) * (sigma*u + beta*u**2) * (rij/rij_norm)
                    vel[ii] -= I/2
                    vel[jj] += I/2
                
    return vel
    
    
h=20    
    
# set the base params
n = 1000 # nb particule
width = 400
height = 400
edge = h
xlim = [edge, width - edge]
ylim = [edge, height - edge]


"Kernel definition"

reg_poly6 = 4/(np.pi*h**8)
reg_spiky = -30/(np.pi*h**5) # for grad kernel
reg_visc = 20/(3*np.pi*h**5) #for laplacian

xyz= []
sphash = Hash(h)
# 
# set-up a dam-break scenario
i = 0
dam_ylim = (ylim[0], ylim[1]-1 )
dam_xlim = (int((xlim[1] - xlim[0]) * 0.0001), int((xlim[1] - xlim[0]) * 0.4))

"dam break"
# for y in range(*dam_ylim, int(h/4)):
#     for x in range(*dam_xlim, int(h/4)):
#         if i < n:
#             xyz.append( [x + np.random.uniform(0, h * 0.1)+h, y + np.random.uniform(0, h * 0.1)]) # x+h to avoid pre clumping at the border bc if not x=0 &x=h clump together bc of BC
#         i += 1
        
#     pass

"droplet fall"
n=2000
dam_ylim =(edge, int(height/4))
dam_xlim =(edge, width-edge)

for y in range(*dam_ylim, int(h/4)):
    for x in range(*dam_xlim, int(h/4)):
        if i < n:
            xyz.append( [x + np.random.uniform(0, h * 0.1), y + np.random.uniform(0, h * 0.1)]) # x+h to avoid pre clumping at the border bc if not x=0 &x=h clump together bc of BC
        i += 1
        
    pass
r = 30
x0 , y0 = width/2, height/2
dam_ylim =(edge, int(height))

for y in range(*dam_ylim, int(h/4)):
    for x in range(*dam_ylim, int(h/4)):
        if ((x-x0)**2 + (y-y0)**2)<r**2:
            if i < n:
                xyz.append( [x + np.random.uniform(0, h * 0.1), y + np.random.uniform(0, h * 0.1)]) # x+h to avoid pre clumping at the border bc if not x=0 &x=h clump together bc of BC
            i+=1
            
xyz = np.asarray(xyz)

for ii in range(xyz.shape[0]):
    sphash.add(xyz[ii],ii)

borderhash = Hash(h)  

border_particle = []

border_ylim = (0, height)
border_xlim = (0, width)

for x in range(*border_xlim, int(h/4)):
    for y in range(0,h, int(h/4)):
        border_particle.append( [x , 0+y ]) 
        border_particle.append( [x , height-y ]) 

for y in range(*border_ylim, int(h/4)):
    for x in range(0,h, int(h/4)):
        border_particle.append( [0+x , y ])         
        border_particle.append( [width-x , y ])         

border_particle = np.asarray(border_particle)

for ii in range(border_particle.shape[0]):
    borderhash.add(border_particle[ii],ii)
    
n_part=xyz.shape[0]

fig, ax1 = plt.subplots()
im = ax1.scatter(xyz[:,0],xyz[:,1])

xmin,xmax = xlim[0], xlim[1] #0,2*width
ymin, ymax = ylim[0], ylim[1] #0,2*height

ax1.set_xlim([xmin-h,xmax+h])
ax1.set_ylim([ymin-h,ymax+h])

plt.show()

m = np.ones(n_part)*5

rho=np.zeros(n_part) # density..
press=np.zeros(n_part) # pressure
vel=np.zeros([n_part,2]) # velocity
dvel=np.zeros([n_part,2]) # velocity correction from xsph
acc_old= np.zeros_like(vel)
cs =np.zeros_like(vel)

g =  -.01 #gravity
mu = 1 # viscosity
dt = 2
damp = -1
f_grav =  np.array([0, g])

gamma = 7
B = 0.1


beta=.05 # quadratic
sigma=0.0001 #linear
k = 0.004 # constante gaz
k_near = .01 # constante gaz
rho0 = 10

niter=2500
#  Dt=1 (wherethetimeunitis1=30second),
#  rho0=10,
#  k=0:004,
#  knear=0:01,
#  kspring=0:3,
#  and alpha=0:3. 

record = True

#%%

def update(tt):
    global xyz, vel, sphash, borderhash, border_particle
    # apply gravity    
    
    vel += dt * f_grav
    
    # apply viscosity
   
    vel = viscosity_impulse(xyz,vel, sigma, beta, dt, h, sphash)

                    
    # for each particle:

    xyz_prev = np.copy(xyz)

    xyz += dt*vel
    
    "update spatial through recomputing it"           
    sphash.cell.clear()
    for ii in range(n_part):
            sphash.add(xyz[ii],ii)
    
    xyz = double_density_relaxation(xyz,rho0, dt, h, k, k_near, sphash, borderhash,border_particle)
     
    # for each particle:
    vel = (xyz - xyz_prev) / dt
    
    jnk = np.argwhere((xyz[:,0]<xmin))# => u=-u
    if jnk.size !=0:
        vel[jnk,0] = damp*vel[jnk,0]
        xyz[jnk,0] = xmin + np.random.rand(jnk.size,1)*xmin*0.001 # add small jitter to avoir particle superposition in corner

    jnk = np.argwhere((xyz[:,0]>xmax))# => u=-u
    if jnk.size !=0:
        vel[jnk,0] = damp*vel[jnk,0]
        xyz[jnk,0] = xmax + np.random.rand(jnk.size,1)*xmax*0.001
        
    jnk = np.argwhere((xyz[:,1]<ymin) ) #=> v=-v   
    if jnk.size!=0:
        vel[jnk,1] = damp*vel[jnk,1]
        xyz[jnk,1] = ymin  + np.random.rand(jnk.size,1)*ymin*0.001
        
    jnk = np.argwhere((xyz[:,1]>ymax) ) #=> v=-v   
    if jnk.size!=0:
        vel[jnk,1] = damp*vel[jnk,1]
        xyz[jnk,1] = ymax + np.random.rand(jnk.size,1)*ymax*0.001
        
    im.set_offsets(xyz)
    
    ax1.set_title(tt)


                
    if tt%50 == 0:
        print(tt)

    return im

ani = animation.FuncAnimation(fig, update,niter)

if record:
    writer = animation.writers['ffmpeg'](fps=30)
    ani.save('demo_sph_clavet.mp4',writer=writer)