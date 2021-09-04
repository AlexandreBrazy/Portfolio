# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 17:40:25 2021

@author: Alexandre Brazy
"""
"""

The code could be better writen

The inverted pendulum is a classic problem as it is:
    - unstable
    - non linear
    - non-underactuated

Several variation exist: kapistza (up/down motion), circular motin, compound and multiple pendulum, control en the position of the cart...

Therefore, it is a good pratical example for control algorithm.
"""

"""
To get the dynamics of the inverted pendulum, we For the are using the Lagrange method.

We set up:
    - x the position of the cart and theta the angle between the pendulum and the upward vertical axis as the global variable
    - mc, mp and l as the mass of the cart, the mass of the pendulum and the length of the pendulum
    - F/ u the control action
    - xp, yp position of the pendulum

xp = x - l sin theta => dxp = xdot - l theta dot cos theta
yp =  l cos theta => -l theta dot sin theta

T = 1/2 mc x_dot**2 + 1/2 mp(xp_dot**2+yp_dot**2)

V = m g l cos theta 

then classic lagrange eq with Qx = F and Qtheta=0

which lead to 

(mp+mc)ddx - mp l ddt ct + mp l dt**2 dt = F
l*ddt- ddx ct - g st = 0

ddx = ( F - dth**2 * l * mp * np.sin(th) + g * mp * np.cos(th) * np.sin(th) ) / ( mc + mp - mp * np.cos(th)**2 )

ddtheta = ( g * (mp + mc) * np.sin(th) + np.cos(th) * ( F - dth**2 * l * mp * np.sin(th)) ) / ( l * (mp + mc - mp * np.cos(th)) )

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mplp
import tkinter as tk
import scipy as sp
from scipy import linalg
from scipy import signal
from scipy import optimize

"physic model"

def physics(x, th, dx, dth,F):
    global g, l, mp, mc, dt
    
    ddx = ( F - .1*dx - dth**2 * l * mp * np.sin(th) + g * mp * np.cos(th) * np.sin(th) ) / ( mc + mp - mp * np.cos(th)**2 ) 
    
    ddth = ( g * (mp + mc) * np.sin(th) + np.cos(th) * ( F - .1*dx - dth**2 * l * mp * np.sin(th)) ) / ( l * (mp + mc - mp * np.cos(th)**2) ) - .2*dth
    
    dx += ddx * dt
    
    dth += ddth * dt
    
    x += dx * dt
    
    th += dth * dt
    return x, th, dx, dth


def PSO():#x, dx, th, dth, error_x, error_th, error_x_cumul, error_cumul, part, pbest, gbest, pbestcoef, v, niter, k): # particle swarm optimization to tune the PIDs
    niterpso = 100
    k=1
            
    nbpart = 2500
    part = np.random.uniform(-250,250, (nbpart,6)) # 100 particles with 6 coefficients, 3 for the PID on the angle, 3 for the PID on position
        
    pbest = np.inf * np.ones(nbpart)
        
    gbest = np.inf # global best as + infinity
    
    pbestcoef = part
    
    tempgbest = []
    v = np.random.uniform(-1,1, size = (nbpart,6))
    while k<niterpso:
        # init/reinit error and init value
                
        th = np.ones(nbpart) * np.deg2rad(15) #+ np.random.randn() # 0 = vertical pendulum
        dth = np.zeros(nbpart)
        
        x = np.ones(nbpart)
        dx = np.zeros(nbpart)
        # init particle, persnnal best and global best.
        
        
            
        """ """

        # F = 000
        dt = 0.01
        
        th_set = 0
        x_set = 0
        
        error_th = th_set - th
        error_cumul = (th_set - th)
        
        error_x = x_set - x
        error_x_cumul = (x_set - x)

        # init particle, pe
        # evaluate for each particle its fitness after 5s
        # 1 - compute the input due to the PID

        
        fitness = np.zeros(nbpart)
        
        
            
        
        for _ in range(500):            
            th = ( th + np.pi) % (2 * np.pi ) - np.pi

            derror = error_th 
            derror_x = error_x             
            
            error_th = th_set - th
            
            error_x = x_set - x            
            F = PID(part[:,0], part[:,1], part[:,2], error_th, derror, dt, error_cumul) # angle
            
            F += PID(part[:,3], part[:,4], part[:,5], error_x, derror_x, dt, error_x_cumul) # position
                 
            F=np.clip(F, -50,50)
            
            x, th, dx, dth = physics(x, th, dx, dth, F)
                        
            # si angle > pi => -2pi
            # si angle <-pi =? +2 pi
            # double pi = Math.PI;
            # while (angle > pi)
            #     angle -= 2 * pi;
            # while (angle < -pi)
            #     angle += 2 * pi;
            
            
            error_cumul += (th_set - th)
            
            error_x_cumul = (x_set - x)
            
            # 2 - Compute the position and the angle
            
            
                
            w1, w2, w3 = 30, 10, 10 # weigth for the fitness, higher mean it should be minize first
        
            fitness += w1 * error_th**2 + w2 * error_x**2 + w3 * F**2
        
        # update the particle with PSO
        
        for ii in range(nbpart):
            if fitness[ii] < pbest[ii]: 
                pbest[ii] = fitness[ii]
                pbestcoef[ii] = part[ii]
            
        jnk = np.argmin(pbest) # global best is the min of all personnal best
        if pbest[jnk] < gbest :
            gbest = pbest[jnk]
            gbestcoef = pbestcoef[jnk]
        
        tempgbest.append(gbest)
        
        Wmax, Wmin, kmax= 1, 0, niterpso
        
        k+=1
        
        W = Wmax - (Wmax - Wmin) / kmax * k
        
        ww, ww2 = 2, 2
        
        v = (W * v 
             + ww * np.random.uniform(size=(nbpart,6)) * (pbestcoef - part)  
             + ww2 * np.random.uniform(size=(nbpart,6)) * (gbestcoef - part) )
        
        part += v
        part = np.clip(part,-250,250)
        # print(part[0:5,:])
    print(gbest, gbestcoef)
    plt.figure()
    plt.plot(tempgbest)
    return(gbestcoef) #,gbest )


def PID(Kp, Ki, Kd, error, old_error,dt, error_sum): #, pos, old_pos):
    
    Kpid = Kp*error + Kd*(error-old_error)/dt + Ki*error_sum*dt # derivative on error => kick when command change
    # Kpid = Kp*error - Kd*(pos-old_pos)/dt + Ki*error_sum*dt # derivative on measurement => way less kick
    return Kpid

def LQR():
    # 1 linearize around equilibrium
    """
    we startt with dx = f(x)
    So the basic idea is to do a taylor development around the equilibrium point
    dx = f(xeq,u) + Df/dx at eq* (x-xeq) + ignore hogh order
    f(xeq,u) = 0 by def (equilibirum point)
    and DF/dx is the jacobian
    we end up with 
    dx = jacobian * delta x => dx = A*x
    also we use small angle approximation
    
    here x = [x, dx, th, dth]
    and dx = A@x + B@F
    """   
    A = np.array([[0, 1, 0, 0],
                  [0, 0, g*mp/mc, 0],
                  [0, 0, 0, 1],       
                  [0, 0, g* (mp+mc)/(l*mc), 0]
                    ])
    
    B = np.array([[0],
                  [1/mc],
                  [0],
                  [1/(l*mc)]])
    
    # C = np.array([[1,0,0,0],
    #               [0,0,1,0]])
    #     Q same as A
    # R nb input x nb input
    Q = np.array([[10, 0, 0, 0],
                  [0, 50, 0, 0],
                  [0, 0, 500, 0],       
                  [0, 0, 0, 250]
                    ])
    
    R = .1
    
    # 2 P Q R S
    
    # 3 ricatti eq
    
    P_lqr = sp.linalg.solve_continuous_are(A,B,Q,R) # solve algebraic ricatti equation 


    K = 1/R * B.transpose() @ P_lqr #compute K
    return K

def Kalman(x, dx, th, dth, F):
    global P,jnkx,jnkkx,jnknx
    # state = np.array([x, dx, th, dth]) 
    state = np.array([x, dx, th, dth]) + np.random.normal(0,1,4) # add some Addititve White Gaussian Noise (AWGN)
    state=state.reshape([4,1])
    
    
    jnkx.append(th)    
    jnknx.append(state[2])
    
    A = np.array([[0, 1, 0, 0],
                  [0, 0, g*mp/mc, 0],
                  [0, 0, 0, 1],       
                  [0, 0, g* (mp+mc)/(l*mc), 0]
                    ])
    
    B = np.array([[0],
                  [1/mc],
                  [0],
                  [1/(l*mc)]])
    
    # project the state ahead
    y = A @ state + B * F
    
    "R uncertainty in the measurement, P uncertainty in the state Q represent noise "
    
    # project the error covariance ahead ie estimate how much noise will be in measurement

    Q = np.eye(4)*5
    P = A @ sp.linalg.inv(P) @ A.transpose() + Q
    # P error value
    # Q covarianceof noise describes distribution of noise
    # init value of Q std of sensor noise given by manufacturer
    # too big or too small its willl diverge
    
    # compute kalman gain
    H = np.array([[1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0],
                  [0,0,0,1]])
   
    R = np.eye(4)*.1
    K = P @ H.transpose() @ sp.linalg.inv(H @ P @ H.transpose() + R)
    # k kalman gain how much we trust this sensor
    # Pk predicted error covarianceH the model of how the sensor reading 
    # reflect the vehicle state
    # R describes the noise un sensor measurement starting value like Q
    
    # update the estimate with measurmenet from sensor
    z = np.array([x,dx,th,dth]).reshape([4,1])
    y = y + (K@(z - H@y))
    # Zk measurement of senser
    
    # update the error covariance
    P = (np.eye(4) - K @ H) @ P
    
    # print(x-y[0])
    jnkkx.append(y[2])
    

    return (y[0].item(), y[1].item(), y[2].item(), y[3].item() ) # quite ugly, y[n] get a singleton array, .item() extract the value
        
def prediction(ss, u, t, x0): #u = control , x0 initila state
    up = np.concatenate([u, np.repeat(u[-1], Np - Nc)]) # extend control for prediction horizon, we optimize for control horizon but need more value for prediction
    
    t, yy, xx = sp.signal.lsim(ss, up, t, x0)    
    
    xx[:,2] = ( xx[:,2] + np.pi) % (2 * np.pi ) - np.pi

    return (yy, xx, up)

def error(u):
    global ss, t, x0, x_set, th_set
    yy, xx, up = prediction(ss, u,t, x0)

    r=[x_set, 0, th_set, 0]
    # np.sum((r - yy)**2)
    xx = xx-r    
    # xx[:,2] = (  xx[:,2] + np.pi) % (2 * np.pi ) - np.pi

    Q = np.array([[10, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 50, 0],       
                  [0, 0, 0, 1]
                    ])
    
    R = .0001
    return np.sum(np.einsum('ij,jk,ji->i',xx,Q,xx.transpose())) + np.sum(up * R * up) # einsum to have the good matrix multiplication

   
    
jnkx=[]
jnkkx=[]
jnknx=[]

jnk=[]
jnkth=[]

def animate(time):
    global x, th, dx, dth, th_set, error_th, dt, error_cumul # to be change in class 
    global x_set, error_x, error_x_cumul, gbestcoef,F, jnkx,jnkkx,kx, kth    
    global ss, t, x0,u, Np, Nc,jnk,jnkth,F
    
    # th = ( th + np.pi) % (2 * np.pi ) - np.pi

    derror = error_th 
    derror_x = error_x 
    error_th = (th_set - th)
    error_x = x_set - x

    error_cumul += (th_set - th)
    
    error_x_cumul = (x_set - x)
    jnk.append(x)
    jnkth.append(th)
    
    "PID"
    # F = PID(240, 1, 30 , error_th, derror, dt, error_cumul)
    
    # F += PID(-2.5, -1,- 2.5 , error_x, derror_x, dt, error_x_cumul)

    # F = PID(gbestcoef[0], gbestcoef[1], gbestcoef[2] , error_th, derror, dt, error_cumul)
    
    # F += PID(gbestcoef[3], gbestcoef[4], gbestcoef[5], error_x, derror_x, dt, error_x_cumul)
    
    "LQR/ LQG"
    "LQG is basically a kalman filter + lqr controller"
    "Control using the kalman estimate, physics using real position"
    # K = LQR()
    kx, kdx, kth, kdth = Kalman(x, dx, th, dth, F)

#     F = K@ np.array([x_set - kx, -kdx, th_set - kth, -kdth]) # -K@(x-xset)
#     # F = K@ np.array([x_set - x, -dx, th_set - th, -dth]) # -K@(x-xset)
#     # print([x, dx, th, dth],'\n',[kx, kdx, kth, kdth] ,'\n \n')
#     # print( 'F',F,"Fx", Fx)
#     F = np.clip(F,-75,75)
        
    x0=np.array([kx, kdx, kth, kdth])
    
    "MPC control every 10 steps of simulation to model the delay of an inline mpc"
    
    if time%10==0:
        bound = sp.optimize.Bounds(-75,75) 
        result = sp.optimize.minimize(error, u,  bounds=bound)
        F = result.x[0]
        u = result.x
    
    x, th, dx, dth = physics(x, th, dx, dth, F) # F[0] when using lqr
    xp = x - l * np.sin(th)
    yp = l * np.cos(th)
    
    ax.set_title('time = {t:.2f} s, NP = {Np}, Nc = {Nc}'.format(t=time*dt, Np=Np,Nc = Nc)) 
    rect.set_xy((x-1, -.5))
    circle.set_center((xp,yp))
    line.set_data([x,xp], [0,yp])
    
    return x, th, dx, dth


th = np.deg2rad(25) #+ np.random.randn() # 0 = vertical pendulum
dth = 0

x = 0
dx = 0

x0 = np.array([x, dx, th, dth])

kx=x+np.random.randn()
kth=th+np.random.randn()*.5
# # for pso

# th = np.ones(100) * np.deg2rad(20) #+ np.random.randn() # 0 = vertical pendulum
# dth = np.zeros(100)

# x = np.zeros(100)
# dx = np.zeros(100)
# # init particle, persnnal best and global best.

# part = np.random.uniform(-100,100, (100,6)) # 100 particles with 6 coefficients, 3 for the PID on the angle, 3 for the PID on position

# pbest = np.inf * np.ones(100)

# gbest = np.inf # global best as + infinity

# pbestcoef = part

# k=0

# v = np.random.uniform(100)*2
    
""" """
l = 4
mp = 1
mc = 5
F = np.zeros(1)
g = 9.81
dt = 0.1
niter = 100

th_set = 0
x_set = 5

error_th = th_set - th
error_cumul = (th_set - th)

error_x = x_set - x
error_x_cumul = (x_set - x)

P = np.eye(4)*10

fig = plt.figure()
ax = fig.add_subplot()
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
# ax.axis("equal")
ax.set_xlim(-10,10)
ax.set_ylim([-l*1.5,l*1.5])

tempx=[]
tempth=[]


xp = x - l * np.sin(th)
yp = l * np.cos(th)


rect = ax.add_patch(mplp.Rectangle((x-1,-.5), 2, 1, color="red")) # bottom left
circle = ax.add_patch(mplp.Circle((xp, yp), 0.4)) # center
line, = ax.plot([x,xp], [0,yp], color="black")

K = LQR()
# gbestcoef = PSO() # particle swarm optimization to tune the PIDs

# for MPC

A = np.array([[0, 1, 0, 0],
              [0, 0, g*mp/mc, 0],
              [0, 0, 0, 1],       
              [0, 0, g* (mp+mc)/(l*mc), 0]
                ])

B = np.array([[0],
              [1/mc],
              [0],
              [1/(l*mc)]])

C = np.array([[1,0,0,0],
              [0,0,1,0]])
D = np.zeros([2,1]) 


ss = sp.signal.StateSpace(A,B,C,D)

x0=np.array([x,dx,th,dth]) #(x,dx,th,dth)


Nc = 8 # control horizon
Np = 60 #prediction horizon

u = np.ones(Nc) #initial guess for control

t = np.arange(0,Np*dt, dt) 
# niter=500


# 
ani = animation.FuncAnimation(fig, animate, niter, repeat=False)

writer = animation.writers['ffmpeg'](fps=10)
ani.save('inverted_pendulum.mp4',writer=writer)
