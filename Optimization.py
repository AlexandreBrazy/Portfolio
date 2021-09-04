# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 23:00:30 2021

@author: Alexandre Brazy
"""

"""
Implementation of the 3 major optimization algorithm
    - Simulated annealing
    - Genetic Algorithm
    - Particle Swarm Optimization
    
They will be test on https://en.wikipedia.org/wiki/Test_functions_for_optimization


"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

def objective_function(x, y):
    "2D - Rastrigin function"
    "A = 10 ; n = n"
    A = 10
    n = 2
    f = A*n + x**2 - A*np.cos(2*np.pi*x) + y**2 - A*np.cos(2*np.pi*y)
    
    "Sphere function"
    # f= x**2+y**2
    return f
    
  
# def neighbor():
#     s
    
# def simulated_annealing():
#    s 

class genetic_algorithm:
    
    def __init__(self, population_size, func):
        self.agent = np.random.uniform(low=xy_min, high=xy_max, size=(npart,2))
        self.func = func
3) Untill convergence repeat:
      a) Select parents from population
      b) Crossover and generate new population
      c) Perform mutation on new population
      d) Calculate fitness for new population
        
    def fitness(self):
        self.fitness = self.func(self.agent[:,0], self.agent[:,1])
    
    def selection():
    
    def crossover():
        
    def mutation:():
        


def particle_swarm_optimization(func, part, vel, personnal_best_pos, personnal_best_fitness, global_best_pos, global_best_fitness, ii, niter):
    
    """ 
    perform one iteration of the particle swarm optimization
    can easily be modified to perform the whole loop    
    but i keep it that way for animation purpose
    
    one note on the parameter.
    
    inertia is in range(0, 1). at 1 => maximum exploration, at 0 maximum exploitation (local search)
    one strategy is to gradually decrease the inertia like temperature in simulated annealing, in order to maximize exploitation at the end
      
    c1 = 0 => no exploration, everything converge to the global best. there a risk of getting stuck in local extrema
    
    c2 = 0 => the particules stuck to their personnal best
    
    the trick is to balance every parameter
    
    """
    global xy_min, xy_max
    inertia = 1 - (ii)/niter
    c1 = 1.5
    c2 = 2
    
    " compute velocity "
    
    vel = inertia * vel + c1 * np.random.uniform(0, 1, size = (len(part),1) ) * (personnal_best_pos - part) + \
                                                 c2 * np.random.uniform(0, 1, size = (len(part),1) ) * (global_best_pos - part)
    
    " update position "
    
    part = part + vel  # to limit speed
    
    " redraw a particle if one get outside "
    " we want the that are not inside the range. ~= not opertor" 
    pos_x = np.argwhere( ~( (part[:,0]>xy_min) & (part[:,0]<xy_max)) ) 
    pos_y = np.argwhere( ~( (part[:,1]>xy_min) & (part[:,1]<xy_max)) )
    
    part_out = np.unique( np.concatenate( (pos_x, pos_y) ) )
    
    " if pos_x and pos_y are empty (size==0) pass, else redraw"
    
    if part_out.size==0:
        pass
    
    else:
                
        part[part_out] = np.random.uniform(low=xy_min, high=xy_max, size=(part_out.size,2))    
                
        " velocity"
        
        vel[part_out] = np.random.uniform(low=xy_min/5, high=xy_max/5, size=(part_out.size,2))
        
    
    " update personnal_best and global_best"
    
    tmp = func(part[:,0], part[:,1]) # compute new fitness

    jnk = np.argwhere(tmp < personnal_best_fitness)
    personnal_best_fitness[jnk] = tmp[jnk]
    personnal_best_pos[jnk] = part[jnk]
        
    tmp = np.argmin(personnal_best_fitness)
    
    if (personnal_best_fitness[tmp] < global_best_fitness):
        global_best_fitness = np.min(personnal_best_fitness)
        global_best_pos = personnal_best_pos[tmp]
   
 
    return part, vel, personnal_best_pos, personnal_best_fitness, global_best_pos, global_best_fitness, inertia


    
x = np.arange(-5, 5, 0.01)

y = np.arange(-5, 5, 0.01)

xx, yy = np.meshgrid(x, y, sparse=True)

z = objective_function(xx,yy)


fig = plt.figure()
ax = fig.add_subplot()
h = ax.contourf(x, y, z)

ax.axis('scaled')


" PSO parameter "

niter = 200 # number of iteration
ii = 0
npart = 20 # number of particle
# for rastrigin
xy_min = -5
xy_max = 5

" pso initialization "

" position "

part = np.random.uniform(low=xy_min, high=xy_max, size=(npart,2))    

# plot = ax.scatter(part[:,0],part[:,1], color = 'r')

plt.title('PSO iteration = {it:.2f}, inertia {inertia}'.format(it=ii, inertia=1))

" velocity"

vel = np.random.uniform(low=xy_min/5, high=xy_max/5, size=(npart,2))

" pbest " 

pbest_fitness = objective_function(part[:,0], part[:,1])
pbest_pos = part

" gbest "

gbest_fitness = np.min(pbest_fitness)
gbest_pos = part[np.argmin(pbest_fitness)]

" PSO loop"

jnk=[]
jnk_inertia=[]
jnkbest=[]

while ii<niter:
    
    part, vel, pbest_pos, pbest_fitness, gbest_pos, gbest_fitness, inertia  =\
            particle_swarm_optimization(objective_function, part, vel, pbest_pos, pbest_fitness, gbest_pos, gbest_fitness, ii, niter)
        
    ii +=1
    

    jnk.append(part.copy())
    jnk_inertia.append(inertia)
    jnkbest.append(gbest_fitness)

    
import matplotlib.animation as animation

def animate(i):
    
    plot.set_offsets(jnk[i])
    ax.set_title('PSO iteration = {it:.2f}, inertia {inert:.2f}'.format(it=i, inert=jnk_inertia[i]))

    
plot = ax.scatter(part[:,0],part[:,1], color = 'r')

# plt.figure()
# plt.plot(jnkbest)
print(gbest_pos)

ani = animation.FuncAnimation(fig, animate, niter, repeat=False)

# writer = animation.writers['ffmpeg'](fps=30)
# ani.save('pso.mp4',writer=writer)
# plt.scatter(part[:,0],part[:,1], color = 'k')
# print(gbest_pos)
    
    
    
    
    

