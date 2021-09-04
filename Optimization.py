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
    
#%%  


class simulated_annealing:
    
    def __init__(self, temperature, population_size, func, xy_min, xy_max):
        self.agent = np.random.uniform(low=xy_min, high=xy_max, size=(population_size,2)).squeeze()
        self.func = func
        self.temperature_init = temperature
        self.temperature = temperature
        self.best = self.agent
    
    def fitness(self,agent):
        "distance between a point and a sqiare: https://stackoverflow.com/questions/5254838/calculating-distance-between-a-point-and-a-rectangular-box-nearest-point"

        dx = max(xy_min - agent[0], 0, agent[0] - xy_max)
        dy = max(xy_min - agent[1], 0, agent[1] - xy_max)
        jnk = np.sqrt(dx*dx + dy*dy) # compute the distance between a point and the border, return 0 if point inside the searchspace

        return self.func(agent[0], agent[1]) + jnk*10000 # we penalize heavyly outside solution
    
    def neighbourhood(self, xy_min, xy_max):
        "change the range of search based on temperature, the lower temp the narrower search"
        self.neighbour = self.agent + (self.temperature/self.temperature_init) * np.random.uniform(low=xy_min, high=xy_max, size=(1,2)).squeeze()
        "we clip the solution in order to stay in the search space, maybe better to penalize outside point ?"
        
    def cooling(self, xy_min, xy_max, _iter, iter_max):
        
        "cooling scheme: http://www.scielo.org.mx/pdf/cys/v21n3/1405-5546-cys-21-03-00493.pdf"
        
        self.temperature = self.temperature_init * ((iter_max -_iter)/iter_max) # linear cooling
        # self.temperature = self.temperature_init * 0.99**_iter # geometrical cooling
        
        self.neighbourhood(xy_min, xy_max)
        
        new_point_fitness = self.fitness(self.neighbour)
        current_fitness = self.fitness(self.agent)
        
        
        "if new state is better than the previous one, we accept it"
        "if not, we accept it nontheless with a proba of exp((-fitness_new-fitness_old)/temperature)"
        if new_point_fitness < current_fitness:
            self.agent = np.copy(self.neighbour)
                       
            
        elif np.random.uniform(0, 1 ,1) < np.exp(-(new_point_fitness - current_fitness) / self.temperature):
            
            self.agent = np.copy(self.neighbour)
            
        "add some memory for the best point visited"            
        if new_point_fitness < self.fitness(self.best):
            self.best = np.copy(self.neighbour) 
#%%
class genetic_algorithm:
    
    def __init__(self, population_size, func, xy_min, xy_max):
        self.agent = np.random.uniform(low=xy_min, high=xy_max, size=(population_size,2))
        self.func = func
        
    def fitness(self):
        return self.func(self.agent[:,0], self.agent[:,1])
    
    def selection(self, k_participant, winner_probability, nb_of_parent):
        
        """
        Tournament Selection
        1- Choose K random participants from the population
        2- Select the winner with a probability p for the best fitness, p*(1-p) for the second best, p*(1-p)**2 for the third and so on
        3- Repeat to get the second parent
        4- Repeat until you have N parents pairs
        
        """
        
        winner_p = [winner_probability * (1 - winner_probability)**_k for _k in range(k_participant) ]
        winner_p[-1] += 1-np.sum(winner_p) # bc we need a total proba of 1
        
        parent = []
        
        for ii in range(nb_of_parent):
            
            "For parent 1"
            
            tmp = np.random.choice(self.agent.shape[0], k_participant, replace = False) # give the row index of the participant, self.agent.shape[0] give the nb of row, replace=false to not get 2 of the same

            participant = self.agent[tmp] # select the participant

            jnk = np.argsort( self.fitness()[tmp] ) # get the index of the sorted the participant based on the fitness 

            winner_1 = np.random.choice(participant.shape[0], 1, replace = False, p = winner_p) # select a winner 
                    
            "For parent 2"
            
            tmp = np.random.choice(self.agent.shape[0], k_participant, replace = False) # give the row index of the participant, self.agent.shape[0] give the nb of row, replace=false to not get 2 of the same

            participant = self.agent[tmp] # select the participant
            
            jnk = np.argsort( self.fitness()[tmp] ) # get the index of the sorted the participant based on the fitness 
            
            winner_2 = np.random.choice(participant[jnk].shape[0], 1, replace = False, p = winner_p) # select a winner 
        
            parent.append([ participant[jnk[winner_1]], participant[jnk[winner_2]] ])
            
        return np.asarray(parent).squeeze()
    
    def crossover(self, parent):
        " BLX-a // blended crossover alpha "
        
        child = []
        
        a = 0.5
        
        for tmp in parent:
            
            mini = np.minimum(tmp[0], tmp[1]) - a*np.abs(tmp[0]-tmp[1])
            maxi = np.maximum(tmp[0], tmp[1]) + a*np.abs(tmp[0]-tmp[1])
            spawn = np.random.uniform(low=mini, high=maxi, size=(1,2))

            child.append(spawn)   

        " BLX-ab // blended crossover alpha beta "
        
        self.agent = np.asarray(child).squeeze()


        
    def mutation(self, mutation_chance, coef_iter):
        
        "coef_iter is (iter_max - iter)/iter_max"
        "Generate a random number"
        
        "if the number is below the mutation chance then either swap gena A and B or generate a new gene for A or B"
        "new gene is generated with a gaussian distribution where the std decreased over time like in simulated annealing"
        for spawn in range(self.agent.shape[0]): 
            tmp = np.random.uniform()
            if tmp < mutation_chance:
                jnk = np.random.uniform()

                if jnk < 0.5: # switch gena
                    tmp_ = np.copy(self.agent[spawn,0])
                    self.agent[spawn, 0] = self.agent[spawn, 1]
                    self.agent[spawn, 1] = tmp_
                elif jnk < 0.75: # change gene A
                    self.agent[spawn, 0] = np.random.normal(self.agent[spawn, 0], coef_iter)
                else:   # change gene B
                    self.agent[spawn, 1] = np.random.normal(self.agent[spawn, 1], coef_iter)
                    
                        
                
        
    
    
#%%        
        


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

#%%
    
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

# " pso initialization "

# " position "

# part = np.random.uniform(low=xy_min, high=xy_max, size=(npart,2))    

# # plot = ax.scatter(part[:,0],part[:,1], color = 'r')

# plt.title('PSO iteration = {it:.2f}, inertia {inertia}'.format(it=ii, inertia=1))

# " velocity"

# vel = np.random.uniform(low=xy_min/5, high=xy_max/5, size=(npart,2))

# " pbest " 

# pbest_fitness = objective_function(part[:,0], part[:,1])
# pbest_pos = part

# " gbest "

# gbest_fitness = np.min(pbest_fitness)
# gbest_pos = part[np.argmin(pbest_fitness)]

" PSO loop"

jnk=[]
jnk_inertia=[]
jnkbest=[]
jnktemp =[]

" Genetic Algorithm "
part=genetic_algorithm(50,objective_function,-5,5)

" Simulated Annealing "

part = simulated_annealing(2, 1, objective_function, -5, 5) # a high init temp will cause everything change to be accepted

niter=2000

while ii<niter:
    "PSO"
    # part, vel, pbest_pos, pbest_fitness, gbest_pos, gbest_fitness, inertia  =\
    #         particle_swarm_optimization(objective_function, part, vel, pbest_pos, pbest_fitness, gbest_pos, gbest_fitness, ii, niter)
     
    "GA"
    # parent = part.selection(5, 0.7, 50) # 5 participant, best fitness win 70% of time, 15 pairs of parents
    
    # part.crossover(parent)
    
    # part.mutation(0.05, (niter-ii)/niter)
    
    # gbest_fitness = np.min(part.fitness())
    
    "SA"
    
    part.neighbourhood(-5,5)
    
    part.cooling(-5,5,ii,niter)
    
    
    
    ii +=1
    
    jnktemp.append(part.temperature)

    jnk.append(part.agent.copy())
    jnk_inertia.append((niter-ii)/niter) #inertia)
    # jnkbest.append(gbest_fitness)
    jnkbest.append(part.fitness(part.best))

    
import matplotlib.animation as animation

def animate(i):
    
    plot.set_offsets(jnk[i])
    ax.set_title('SA iteration = {it:.2f}, temperature = {temp:.4f}'.format(it=i, temp=jnktemp[i]))

    
# plot = ax.scatter(part.agent[:,0],part.agent[:,1], color = 'r') # for PSO and GA
plot = ax.scatter(part.agent[0],part.agent[1], color = 'r') # for SA
print(part.best)
print(part.agent)
plt.figure()
plt.plot(jnkbest)
# print(gbest_pos)

ani = animation.FuncAnimation(fig, animate, niter, repeat=False)

writer = animation.writers['ffmpeg'](fps=60)
ani.save('SA2.mp4',writer=writer)

    
    
    
    
