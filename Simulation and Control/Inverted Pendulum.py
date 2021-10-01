# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 17:40:25 2021

@author: Alexandre Brazy
"""
"""

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
import scipy as sp 
from scipy import linalg # to solve ricatti eq for LQR
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mplp
from scipy import signal
from scipy import optimize



class pendulum:
    
    def __init__(self):
        
        self.x = 0 # oosition
        self.dx = 0 # velocity
        self.th = .2 # angle
        self.dth = 0 # angle velocity
        
        self.F = 0 # control force
        
        self.g = 9.81
        self.mp =  1 # mass of pendulum
        self.mc = 5 # mass of cart
        self.l = 4 # length of the pendulum arm
        
    def physics(self, t, X):
        
        "theta = direct trigo direction"
        
        x, dx, th, dth = X.ravel() # .ravel() to ease unpack
        
        dX = np.array([
                [dx],
    
                [( self.F - dth**2 * self.l * self.mp * np.sin(th) + self.g * self.mp * np.cos(th) * np.sin(th) ) / ( self.mc + self.mp - self.mp * np.cos(th)**2 ) - .1*dx ],
  
                [dth],
                
                [( self.g * (self.mp + self.mc) * np.sin(th) + np.cos(th) * ( self.F - dth**2 * self.l * self.mp * np.sin(th)) ) / ( self.l * (self.mp + self.mc - self.mp * np.cos(th)**2) ) - .2*dth]
               
                 ])
            
        return dX
    
    def rk4(self, t, X, h):
        
        """
        for a coupled 2nd order ode for rk4
        
        1- decompose the system in a coupled 1st order ode
            dy1 = f1
            dy2 = f2
            dy3 = f3
            ...
        
        2- set X = (y1, y2, y3,...) and F = (f1, f2, f3,...)
        
        3- use the classic formula of rk4
        
        """
        
        k1 = self.physics(t, X)
        
        k2 = self.physics(t+h/2, X+h*k1/2)
        
        k3 = self.physics(t+h/2, X+h*k2/2)
        
        k4 = self.physics(t+h, X+h*k3)
        
        X = X + h*(k1 + 2*k2 + 2*k3 + k4)/6    
        
        return X 
    
class control:
    
    def __init__(self, command , controller ):
        
        "command = x, th"
        
        self.command = command

        
        if controller == 'PID':                

            self.err = np.zeros_like(command)
            self.derr = np.zeros_like(command)
            self.int_err = np.zeros_like(command)
        
        elif controller=='LQR':                        

            self.err = np.zeros_like(command)
            
        elif controller=='MPC':
                       
            self.err = 0    
            
        else: print('invalid controller')
            
    def update_err_pid(self, state, dstate, dt):
        
        "state is the process value (eg position) and dstate its derivative (eg velocity)"
        
        self.err = self.command - state 
        # self.derr = (self.err - old_err) / dt
        self.derr = - dstate # we use the derivative on measurement as there is less kick when the command change
        self.int_err = self.int_err + self.err*dt

        
    def PID_init(self, coef):
        """
        For PID we need two instance or a 2d pid. one for angle one for position.
        if the angle is to the left the cart must go left and viceversa so coef are >0
        if the cart is one the left its must go rigth and vice versa so cef are <0
        """
        self.Kp, self.Ki, self.Kd = coef
        
    def PID(self):
        
        Kpid = self.Kp*self.err + self.Kd*self.derr + self.Ki*self.int_err # derivative on measurement => way less kick
   
        return Kpid
        
    def update_err_lqr(self, state):
        
        "state is the state of the system "
        
        self.err = self.command - state # numpy array better ?
  

    def LQR(self, A, B, Q, R):

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
        and dx = A@x + B@F => continuous ricatti eq (continuous_are)
        if X+1 = A@x + B@u => discrete ricatti eq 
        and its this eq to use in kalman
        """   
        
        # 2 P Q R S
        
        # 3 ricatti eq
        
        P_lqr = sp.linalg.solve_continuous_are(A,B,Q,R) # solve algebraic ricatti equation 
    
        "R[:,None] to add one dim so R is square instead of (1,)"
        K = np.linalg.inv(R) @ B.transpose() @ P_lqr # compute K for continuous system
        
        # K for discrete case is compute differently not implemented
        
        
        self.K = K
  
        
    def MPC_init(self, A, B, C, D, Q, R, Nc, Np, u, h):

        self.state_space = sp.signal.StateSpace(A, B, C, D, dt=h)
        self.Q = Q
        self.R = R
    
        self.Nc = Nc # control horizon
        self.Np = Np # prediction horizon
        
        self.u = u # initial guess for mpc control law
        

    def MPC_prediction(self, u, x0, t):
        'u = control , x0 current state'
        'u is usually given as the previous compute control to speed up calculation'
        'The time steps at which the input is defined. If t is given, it must be the same length as u, and the final value in t determines the number of steps returned in the output.'
        
        up = np.concatenate([u, np.repeat(u[-1], self.Np - self.Nc)]) 
        up = up.reshape(up.shape[0],1)
        
        # extend control for prediction horizon, we optimize for control horizon but need more value for prediction horizon
        
        t, yy, xx = sp.signal.dlsim(self.state_space, up, t, x0.squeeze())     # squeeze for dimension, dont really no why
        
        xx[:,2] = ( xx[:,2] + np.pi) % (2 * np.pi ) - np.pi # for the angle value to stay between [-pi ; pi]
    
        return (yy, xx, up)

    def MPC_error(self, u, x0, t):
        "u = input command for the control horizon, Q state penalty, R input penalty"
        
        yy, xx, up = self.MPC_prediction(u, t, x0)
    
        r=[self.command[0], 0, self.command[2], 0]
        # np.sum((r - yy)**2)
        xx = xx-r    
        xx[:,2] = (  xx[:,2] + np.pi) % (2 * np.pi ) - np.pi
    
        return np.sum(np.einsum('ij,jk,ji->i',xx,self.Q,xx.transpose())) + np.sum(up @ self.R @ up.transpose()) # einsum to have the matrix multiplication apply on all row of xx
    
    def MPC_command(self, u, x0, t, bound = None):
    
        
        if type(bound) != tuple: # if bound is not a tuple then no bound is use
            result = sp.optimize.minimize(self.MPC_error, u,args=(x0, t) ) #,  bounds=bound)
        
        else:
            bound = sp.optimize.Bounds(bound[0], bound[1]) 
            result = sp.optimize.minimize(self.MPC_error, u,args=(x0, t), bounds=bound)
        
            
        return result
        


class Kalman:
    
    def __init__(self, state, A, B, Q, H, R, P):

        "to project the system"
        self.A = A
        self.B = B
        self.H = H
        
        "R uncertainty in the measurement, P uncertainty in the state Q represent process noise ie how far the model is from reality "
        
        self.Q = Q
        self.R = R
        
        self.P = P
        
        self.kx = state[0]
        self.kdx = state[1] 
        self.kth = state[2]
        self.kdth = state[3]
        
        
    def kalman_filter(self, kalman_state, measurement, F):
        
        kalman_state = kalman_state.reshape([4,1])
           
        # project the kalman_state ahead
        y = self.A @ kalman_state + self.B * F # normally, B@F but in the present case F is just a value and not a matrix so it become B*F

        # project the error covariance ahead ie estimate how much noise will be in measurement

        
        self.P = self.A @ self.P @ self.A.transpose() + self.Q
        # P error value
        # Q covarianceof noise describes distribution of noise
        # init value of Q std of sensor noise given by manufacturer
        # too big or too small its willl diverge
        
        
        # compute kalman gain
        
        
        
        K = self.P @ self.H.transpose() @ sp.linalg.inv(self.H @ self.P @ self.H.transpose() + self.R)
        # k kalman gain how much we trust this sensor
        # Pk predicted error covarianceH the model of how the sensor reading 
        # reflect the vehicle state
        # R describes the noise un sensor measurement starting value like Q
        
        # update the estimate with measurmenet from sensor
        z = measurement # add some Addititve White Gaussian Noise (AWGN) for position/vel std of .5 for angle/angle vel std of .05
        z = z.reshape([4,1])
        y = y + ( K@(z - self.H @ y) )
        # Zk measurement of senser
        
        # update the error covariance
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        
    
        return y
    
        
pend = pendulum()

t=0
h=0.05

X = np.array([
              [pend.x],
              [pend.dx],
              [pend.th],
              [pend.dth]
              ])

# ctrl = control(np.array([5,0]),'PID') # set the command (as input) and error, command need to be in row format
# ctrl = control(np.array([5,0,0,0]),'LQR') # set the command (as input) and error, command need to be in row format
ctrl = control(np.array([5,0,0,0]),'MPC') # set the command (as input) and error, command need to be in row format


"PID coef: 1st column position, 2nd column angle"
pid_coef = np.array([
                    [-5, 200],
                    [-1, 1], 
                    [-10, 100]
                    ])

"LQR init"
A = np.array([[0, 1, 0, 0],
              [0, 0, pend.g*pend.mp/pend.mc, 0],
              [0, 0, 0, 1],       
              [0, 0, pend.g* (pend.mp+pend.mc)/(pend.l*pend.mc), 0]
                ])

B = np.array([[0],
              [1/pend.mc],
              [0],
              [1/(pend.l*pend.mc)]])

C = np.array([[1,0,0,0],
              [0,0,1,0]])
        
# Q size same as A
# R size nb input x nb input
Q = np.array([[10, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 50, 0],       
              [0, 0, 0, 1]
                ])

R = np.array([.1], ndmin=2) # ndim =2 so we have an array

"MPC init"

Nc, Np = 8, 60

u = np.ones(Nc) #initial guess for control

t_mpc = np.arange(0,Np*h, h) 

C = np.eye(4)

D = np.zeros([4,1]) 

# ctrl.PID_init(pid_coef)
# ctrl.LQR(A, B, Q, R)
ctrl.MPC_init((np.eye(4)+A*h), B*h, C, D, Q, R, Nc, Np, u, h)

"for mpc a time step too small will result in instability unless increasing Nc and Np"

"Kalman filter initialization"

"""
R describes the noise un sensor measurement starting value like Q
Q describes the process noise ie how far your model is from reality
H transform the meausrement into the state space ie electrical current to position for example. here its already in the correct form so its a identity matrix
the smallest the more we trust
"""
Q = np.eye(4)*1e-5
H = np.eye(4)
R = (np.diag([1,1,0.5,0.5])*.4)**2 # np.eye(4)

P = np.eye(4)*2

X = X + np.random.normal(0,[[0.5],[.5],[.05],[.05]],(4,1))

kalman = Kalman(X.ravel(), (np.eye(4)+A*h), B*h, Q, H, R, P) 

# A@x+b@u = dx so to get the next step we compute X+1 = X + dX*dt= X + (Ax + Bu)*dt = (1 + A*dt)@X + B*dt@u, 1 being identity matrix

kX = np.array([kalman.kx, kalman.kdx, kalman.kth, kalman.kdth]) # kalman state

linX = np.copy(X)


"plot initialization"
fig = plt.figure()
ax = fig.add_subplot()
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
# ax.axis("equal")
ax.set_xlim(-10,10)
ax.set_ylim([-pend.l*1.5,pend.l*1.5])
ax.set_facecolor('k')

xp = pend.x - pend.l * np.sin(pend.th)
yp = pend.l * np.cos(pend.th)

rect = ax.add_patch(mplp.Rectangle((pend.x-1,-.5), 2, 1, color="red")) # bottom left
circle = ax.add_patch(mplp.Circle((xp, yp), 0.4)) # center
line, = ax.plot([pend.x,xp], [0,yp], color="w")

C = np.array([[1,0,0,0],
              [0,0,1,0]])
D = np.zeros([2,1]) 
  

#%%

for t in range(500):    
    
    X = pend.rk4(t, X, h)
    pend.x, pend.dx, pend.th, pend.dth = X.ravel() # .ravel() flattened the array to ease the unpacking
    
    pend.th = ( pend.th + np.pi) % (2 * np.pi ) - np.pi # to wrap the angle between [-pi, pi] instead of [0, 2pi]

    "State estimator ie Kalman filter"
    
    measurement = X + np.random.normal(0,[[1],[1],[0.5],[0.5]],(4,1))*.4
    kalman.kx, kalman.kdx, kalman.kth, kalman.kdth = kalman.kalman_filter(kX, measurement, pend.F).ravel()
    kalman.kth = ( kalman.kth + np.pi) % (2 * np.pi ) - np.pi
    kX = np.array([kalman.kx, kalman.kdx, kalman.kth, kalman.kdth]) # kalman state

    "PID control" 
    
    # ctrl.update_err_pid(X.ravel()[[0,2]], X.ravel()[[1,3]], h) # X[[a,b]] to get the value, if X[a,b] => classic indexing
    # pend.F = np.clip( ctrl.PID().sum(), -40, 40) # .item() if we use just one prcess variable to control, .sum if we use the two variable

    "LQR control"
    
    # ctrl.update_err_lqr(kX.ravel()) 
    # pend.F = np.clip( (ctrl.K @ ctrl.err).item(), -40, 40 ) # lqr control = -K@(x-xset) or K@ (x_set-x), err is define as x_set-x

    # measurement[1], measurement[3] = 0,0
  
    "MPC control"    
    """
    on real state works well for dt=.1 Nc = 8 Np = 60
    with estimator dt should be reduce to .05
    """
    if t%2==0: # to simulate lag due to computation
        result = ctrl.MPC_command(u, t_mpc, kX, (-50,50))    
        pend.F = result.x[0]
        u = result.x # to use the current solution as the starting point of the next mpc iteration
            
    
    xp = pend.x - pend.l * np.sin(pend.th)
    yp = pend.l * np.cos(pend.th)
    
    ax.set_title('time = {t:.2f} s'.format(t=t*h)) 
    rect.set_xy((pend.x-1, -.5))
    circle.set_center((xp,yp))
    line.set_data([pend.x,xp], [0,yp])
    plt.pause(.00001)

