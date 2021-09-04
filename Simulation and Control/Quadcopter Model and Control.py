# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 00:11:36 2021

@author: Alexandre Brazy
"""


"""
# https://cyberleninka.org/article/n/689329.pdf
https://www.youtube.com/watch?v=9GZjtfYOXao&list=PLxdnSsBqCrrEx3A6W94sQGClk6Q4YCg-h&index=8

front
	F1
	|
F2--G--F4
	|
	F3
	
       
Back

Using lagrangian mmechanics, 3 ''models'' can be extracted:
    - the simplest one consider a diagonal inertia matrix with small angle approximation making angular speed in body and reference frame equals 
        which lead to phi_ddot = moment_x/Ix
    - a more complete one consider a transformation between the 2 frame for the angular speed but keep the same inertia matrix (or consider it to be still diagonal)
    - the most complete consider the transformation also for the inertia matrix with Ek_rot = 1/2*w_t*I*w = 1/2*w_t*T_t*I*T*w = 1/2*eta*J*eta
        where eta= phi, theta, psi and T the transform matrix and *_t the transpose

We will use the second approximation. the most complete one is too complex, and the simplest one too simple.

Lagrangian = Ek_trans + Ek_rot - Epot
           = 1/2*m*(vx**2 + vy**2 + vz**2) + 1/2 * (Ix*wx**2 + Iy*wy**2 + Iz*wz**2) - m*g*z
    
with I diagonal inertia matrix bc quadcopter is consider symmetrical

vx,vy,vz : velocity of the center of mass in the reference frame
wx, wy, wz : angular velocity in the body frame

http://www.sky-sailor.ethz.ch/docs/Modelling_and_Control_of_the_UAV_Sky-Sailor.pdf

wx = phi_dot - psi_dot*sin(theta)
wy = theta_dot*cos(phi)+psi_dot*sin(phi)*cos(theta)
wz = -theta_dot*sin(phi) + psi_dot*cos(phi)*cos(theta)

also we have
v_ref = Rzyx*v_body

q = [x, y, z, phi, theta, psi]T
q_dot = [x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot]T

d/dt(dL/dq_dot) - dL/dq = F

generalized force
F = [Fx, Fy, Fz, torque phi, torque theta, torque psi]T

For translation:
    only force is the thrust from rotor in the z direction of the body reference frame 
    so F = R.Fbody
    with R global rotation matrix for euler angle
    so F = [-f sin(theta), f sin phi cos theta, f cos phi cos theta]T
    and f = total thrust
    
For rotation:
    We are considering an X quad copter, so some change are made compare to the + model
    mostly, change in the length of arm for torque (diagonal instead of straigth so sqrt(2)/2*l instead of l) and the number of rotor consider for torque
    t = [l*sqrt(2)/2*(F1-F2-F3+F4), l*sqrt(2)/2*(F1+F2-F3-F4), sum(l x Fi)i]

https://www.mtwallets.com/wp-content/uploads/2018/03/PID-vs-LQ-Control-Techniques.pdf

So, apply euler lagrange eq (d/dt(dL/dq_dot) - dL/dq = Generalized Force) we end up wiht:

    q_ddot = [u1/m *(cos(psi)*sin(theta)*cos(phi) + sin(psi)*sin(phi)) -1/m Ax (drag)
              u1/m *(sin(psi)*sin(theta)*cos(phi)-cos(psi)*sin(phi)) - 1/m*Ay
              u1/m *(cos(theta)*cos(phi) - g
              theta_dot*psi_dot*(Iy-Iz)/Ix - J/Ix*theta_dot*omega + u2/Ix
              phi_dot*psi_dot*(Iz-Ix)/Iy + J/Iy*phi_dot*omega +u3/Iy
              phi_dot*theta_dot*(Ix-Iy)/Iz + u4/Iz
              ]
   with u1 = total thrust
		u2 = l *(F3-F1)
		u3 = l * (F4-F2)
		u4 = b *(F1 - F2 + F3 - F4)
the J term is the gyroscopic effect of the propeller    

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.animation as animation
import scipy as sp
from scipy import linalg
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d

def PID(Kp, Ki, Kd, error, old_error,dt, error_sum): #, pos, old_pos):
    
    Kpid = Kp*error + Kd*(error-old_error)/dt + Ki*error_sum*dt # derivative on error => kick when command change
    # Kpid = Kp*error - Kd*(pos-old_pos)/dt + Ki*error_sum*dt # derivative on measurement => way less kick
    return Kpid

def LQR(A, B, Q, R):
    try:
        P = sp.linalg.solve_continuous_are(A,B,Q,R) # solve algebraic ricatti equation 
    except ValueError:
        P = sp.linalg.solve_continuous_are(A,B.transpose(),Q,R) # solve algebraic ricatti equation 

    """
    Try to compute K,
    if error due to size of R (if dim(R)=1 ie a number it cant be inverse using sp/np.linalg)
    then simply divide B@P by R
    """
    try: 
        K = sp.linalg.inv(R) @ B.transpose() @ P #compute K
    except ValueError: #sp.linalg.LinAlgError:
        # Not invertible. Skip this one.
        K = 1/R* B.transpose() @ P #compute K

    return K



def plot_quad(q):
    global l, line1, line2, p     

    
    Rx = np.array([[1,0,0],[0,np.cos(q[3]),-np.sin(q[3])],[0, np.sin(q[3]), np.cos(q[3])]])
    
    Ry = np.array([[np.cos(q[4]),0,np.sin(q[4])],[0,1,0],[-np.sin(q[4]),0,np.cos(q[4])]])
    
    Rz = np.array([[np.cos(q[5]),-np.sin(q[5]),0],[np.sin(q[5]), np.cos(q[5]),0],[0,0,1]])
    
    R=Rz@Ry@Rx
    
    axe1 =np.array([R@[ l*np.sqrt(2)/2,l*np.sqrt(2)/2,0]+q[:3],
                    R@[-l*np.sqrt(2)/2,-l*np.sqrt(2)/2,0]+q[:3] ] )
    # axe1 = rotor 1 - rotor 3
    
    axe2 = np.array([R@[ -l*np.sqrt(2)/2, l*np.sqrt(2)/2, 0]+q[:3],
                    R@[l*np.sqrt(2)/2, -l*np.sqrt(2)/2, 0]+q[:3] ] )
    
    
    line1.set_data_3d(axe1[:,0], axe1[:,1], axe1[:,2])
    line2.set_data_3d(axe2[:,0], axe2[:,1], axe2[:,2])
    
pos = np.array([0, 0, 0], dtype=float) # x, y, z
vel = np.array([0, 0, 0], dtype=float)
angle = np.array([0, 0, 0], dtype=float) # phi, theta, psi ; roll pitch yaw
angle_dot = np.array([0, 0, 0], dtype=float)
m = .5 # masse in kg
g = 9.81
dt = 0.015
Ix = 0.0196
Iy = 0.0196
Iz = 0.0264
l = 0.25 # in meter

q = np.hstack((pos, angle))
q_dot = np.hstack((vel, angle_dot))

omega1=111
omega2=111
omega3=111
omega4=111
k_L=3e-5
k_M=1.1e-6

def F(k_L, omega):
    return k_L*omega**2

def M(k_M, omega):
    return k_M*omega**2

F1 = F(k_L, omega1)
F2 = F(k_L, omega2)
F3 = F(k_L, omega3)
F4 = F(k_L, omega4)

temp = []

J=1e-3
Ax=5e-2
Ay=5e-2
Az=5e-2

tempdot=[]
 
z_command = 1
error_z = z_command - q[2]
old_error_z = error_z
error_sum_z = 0

x_command = 20
error_x = x_command - q[0]
old_error_x = error_x
error_sum_x = 0

y_command = 20
error_y = y_command - q[1]
old_error_y = error_y
error_sum_y = 0

psi_command = 90*np.pi/180
error_psi = z_command - q[5]
old_error_psi = error_psi
error_sum_psi = 0

error_phi =0
old_error_phi = error_phi
error_sum_phi = 0

error_theta =0 
old_error_theta = error_theta
error_sum_theta = 0

niter=30000


Ft =  [F(k_L, omega1), F(k_L, omega2), F(k_L, omega3), F(k_L, omega4)]
W=omega1-omega2+omega3-omega4

u1 = np.sum(Ft)
tx = l*np.sqrt(2)/2*(-Ft[0]-Ft[3]+Ft[2]+Ft[1])
ty = l*np.sqrt(2)/2*(Ft[0]+Ft[1]-Ft[2]-Ft[3])
tz = np.sum([M(k_M, omega1), -M(k_M, omega2), M(k_M, omega3), -M(k_M, omega4)])

# for altitude
A = np.array([[0, 1],
              [0, 0]])
B = np.array([[0],
              [1] ])
# u = [uz]
Q = np.array([[5,0],[0,20]])

R = np.array([.01])

K_alt = LQR(A,B,Q,R)

u2=u3=u4=0

jnk = np.linspace(0,8*np.pi,int(50*16/2+1) )
i=500
r = 5
x = r * np.cos(2*jnk+2*np.pi)
y = r * np.sin(jnk*16)
z=jnk
traj=0
x_command=x[0]
y_command=y[0]
z_command = z[0]
#%%
              
for t in range(niter):

    
    
    W=0
    
	"integration"
    q_ddot = np.array([u1/m *(np.cos(q[5])*np.sin(q[4])*np.cos(q[3]) + np.sin(q[5])*np.sin(q[3])) -1/m*Ax*q_dot[0] , 
              u1/m *(np.sin(q[5])*np.sin(q[4])*np.cos(q[3])-np.cos(q[5])*np.sin(q[3])) - 1/m*Ay*q_dot[1],
              u1/m *np.cos(q[4])*np.cos(q[3]) - g - 1/m*Az*q_dot[2],
              
              q_dot[4]*q_dot[5]*(Iy-Iz)/Ix + l*u2/Ix + J/Ix*q_dot[4]*W, 
              q_dot[3]*q_dot[5]*(Iz-Ix)/Iy + l*u3/Iy - J/Iy*q_dot[3]*W,
              q_dot[3]*q_dot[4]*(Ix-Iy)/Iz + u4/Iz  ])


    q_dot += q_ddot * dt
        
    q += q_dot*dt 
    
        
    temp.append(np.copy(q))
    tempdot.append(np.copy(q_dot))
    
	
	"LQR Control"
    u1 =  K_alt @ (np.array([z_command, 0]) - np.array([q[2], q_dot[2]]) )
    u1=u1[0]
    
    # # for position
    A = np.array([[0, 1, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0] ])
    a= u1 / m
    B = np.array([[0, 0],
                  [a, 0],
                  [0, 0],
                  [0, a] ])
    
    Q = np.array([[50, 0, 0, 0],
                      [0, 100, 0, 0],
                      [0, 0, 50, 0],
                      [0, 0, 0, 100] ])
    
    R = np.array([[.001,0],[0,.001]])
    
    K_pos = LQR(A,B,Q,R)
    
    ux, uy = K_pos@(np.array([x_command-q[0], -q_dot[0], y_command-q[1], -q_dot[1]]) ) # x error, x_dot error, y error, y_dot error

    phi_command = ux
    
    theta_command = uy
    
    if phi_command >0.4: phi_command=0.4
    if phi_command <-0.4: phi_command=-0.4

    if theta_command >0.4: theta_command=0.4
    if theta_command <-0.4: theta_command=-0.4
    
    A = np.array([[0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 0] ])
    
    B = np.array([[0, 0, 0],
                  [1/Ix, 0, 0],
                  [0, 0, 0],
                  [0, 1/Iy, 0],
                  [0, 0, 0],
                  [0, 0, 1/Iz] ])
    
    Q = np.array([[10,0,0,0,0,0],
                  [0,1,0,0,0,0],
                  [0,0,10,0,0,0],
                  [0,0,0,1,0,0],
                  [0,0,0,0,10,0],
                  [0,0,0,0,0,1]])
    
    R = np.array([[.1,0,0],
                  [0,0.1,0],
                  [0,0,0.1]])
    
    K_att = LQR(A,B,Q,R)
        
    u2,u3,u4 = K_att @ np.array([ phi_command- q[3], 0-q_dot[3] , theta_command-q[4] ,0-q_dot[4] , psi_command-q[5] , 0-q_dot[5]])
    u2=u2
    u3=u3
    u4=u4

	"Trajectory"
    if (t%75==0) & (t>000) & ( traj<(len(x)-1)): 
        x_command = x[traj]
        y_command = y[traj]
        z_command = z[traj]
        traj+=1
    

    "PID"
    # error_z = z_command - q[2]
    # error_sum_z += error_z*dt
    
    # u1 = PID(20,10,50,error_z, old_error_z, dt, error_sum_z)
    # old_error_z = error_z
    
    # error_x = x_command - q[0]
    # error_sum_x += error_x*dt
    
    # x_ddot = PID(10,4,100,error_x, old_error_x, dt, error_sum_x)
    # old_error_x = error_x
       
    # error_y = y_command - q[1]
    # error_sum_y += error_y*dt
    
    # y_ddot = PID(10,4,100,error_y, old_error_y, dt, error_sum_y)
    # old_error_y = error_y
    
    # phi_command = m/u1*(np.sin(q[5])*x_ddot + np.cos(q[5])*y_ddot)
    # theta_command = m/u1*(-np.cos(q[5])*x_ddot + np.sin(q[5])*y_ddot)
    
    # if phi_command >0.4: phi_command=0.4
    # if phi_command <-0.4: phi_command=-0.4

    # if theta_command >0.4: theta_command=0.4
    # if theta_command <-0.4: theta_command=-0.4
    
    
    # error_phi = phi_command - q[3]
    # error_sum_phi += error_phi*dt
    # error_theta = theta_command - q[4]
    # error_sum_theta += error_theta*dt
    # u2 = PID(100,0,200,error_phi, old_error_phi, dt, error_sum_phi)
    # u3 = PID(100,0,200,error_theta, old_error_theta, dt, error_sum_theta)
    # old_error_phi = error_phi
    # old_error_theta = error_theta

    # psi_command = np.arccos( np.dot( [x_command-q[0], y_command-q[1]], [1,0] / np.sqrt( (x_command-q[0])**2+ (y_command-q[1])**2) ) )
    
    # error_psi = psi_command - q[5]
    # error_sum_psi += error_psi*dt
    # u4 = PID(300,4,10,error_psi, old_error_psi, dt, error_sum_psi)
    # old_error_psi = error_psi

def animate(t):
    global temp, ax, line_CM
    
    q=temp[t]    
    line_cm.append(q[:3])
    # line_CM.remove()
    jnk=np.asarray(line_cm)
    line_CM.set_data_3d(jnk[:,0],jnk[:,1],jnk[:,2])
    plot_quad(q)
    ax.set_title(t)
    
temp = np.asarray(temp)  
tempdot = np.asarray(tempdot)  
    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plt.plot(temp[:,0], temp[:,1], temp[:,2], label='trajectory')
plt.plot(x,y,z, label='Planned trajectory')
ax.legend()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim([np.min(temp[:,0]), np.max(temp[:,0])])
ax.set_ylim([np.min(temp[:,1]), np.max(temp[:,1])])
ax.set_zlim([np.min(temp[:,2]), np.max(temp[:,2])])

ax.set_box_aspect((1, 1, 1))
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

q=temp[0]

Rx = np.array([[1,0,0],[0,np.cos(q[3]),-np.sin(q[3])],[0, np.sin(q[3]), np.cos(q[3])]])

Ry = np.array([[np.cos(q[4]),0,np.sin(q[4])],[0,1,0],[-np.sin(q[4]),0,np.cos(q[4])]])

Rz = np.array([[np.cos(q[5]),-np.sin(q[5]),0],[np.sin(q[5]), np.cos(q[5]),0],[0,0,1]])

R=Rz@Ry@Rx

axe1 =np.array([R@[ l*np.sqrt(2)/2,l*np.sqrt(2)/2,0]+q[:3],
                R@[-l*np.sqrt(2)/2,-l*np.sqrt(2)/2,0]+q[:3] ] )
# axe1 = rotor 1 - rotor 3

axe2 = np.array([R@[ -l*np.sqrt(2)/2, l*np.sqrt(2)/2, 0]+q[:3],
                    R@[l*np.sqrt(2)/2, -l*np.sqrt(2)/2, 0]+q[:3] ] )


line1, = plt.plot(axe1[:,0], axe1[:,1], axe1[:,2])
line2, = plt.plot(axe2[:,0], axe2[:,1], axe2[:,2])
line_CM, = plt.plot(q[0], q[1], q[2])
line_cm=[q[:3]]  


# p = plt.Circle((axe1[0,0], axe1[0,1]), l/2)
# ax.add_patch(p)
# art3d.patch_2d_to_3d(p, z=q[2], zdir="xy")
  

ani = animation.FuncAnimation(fig, animate, niter, repeat=False)

writer = animation.writers['ffmpeg'](fps=10)
ani.save('demo_drone.mp4',writer=writer)
        