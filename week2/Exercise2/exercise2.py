import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation
from numpy.core.fromnumeric import size
from SimFunctions import SimulationFunctions

#Scripts simulates control of a two-joint arm using a feedback controller model.
#The arm reaches to 4 targets.
#Closed loop: slow movements that require precision
#Open loop: fast movement 


############ Parameters ###########
#Movement duration
T=.6
dt=.01     # Time step
L=6.0      # Simulation duration 600 sek
kp=50.0   # Proportional gain
kd= 5.0     # Derivative parameter (damping)
le1=.3     # Upper arm length
le2=.3     # Lower arm length
m1=3       # Upper arm mass
m2=3       # Lower arm mass
g=-9.8     # Gravity



############# Functions #############
Var = [T,dt,L,kp,kd,le1,le2,m1,m2,g]
Sim = SimulationFunctions(Var)


############# Variables #############
ang=[-np.pi/4, np.pi]               #Joint angles [shoulder elbow] 
ang_rec=np.zeros((int(L/dt+1),2))   

vel=[0, 0]                          #Joint velocity [shoulder elbow]
vel_rec=np.zeros((int(L/dt+1),2))

acc=[0, 0]                          #Joint acceleration [shoulder elbow]
acc_rec=np.zeros((int(L/dt+1),2))
jerk_rec=np.zeros((int(L/dt+1),2))  #Jerk [shoulder elbow]
shoulder_pos=[0, 0]                 #Shoulder position 
elbow_pos=[0, 0]                    #Elbow position
wrist_pos=[0, 0]                    #Wrist position
wrist_pos_rec=np.zeros((int(L/dt+1),2))
init_wrist_pos=wrist_pos 
final_wrist_pos = [[0.3, 0.0],  [0.0, 0.0],  [.3*np.cos(np.pi/4), .3*np.sin(np.pi/4)],  [0.0, 0.0], 
                 [0.0, .3],  [0.0, 0.0],  [.3*np.cos(3*np.pi/4), .3*np.sin(3*np.pi/4)],  [0.0, 0.0]]

curr_target=0     #Current target index
start_t=0         # Movement start_time


############# Task 2.2.1 plotting ##############
desired_ang = np.zeros([int(L/dt),2])      #Total number of timesteps, (2):Two columns, one for albow torque and one for the shoulder torque
desired_torques = np.zeros([int(L/dt),2])  #Total number of timesteps  (2):Two columns, one for albow torque and one for the shoulder torque


################ Task 2.2.2: adding a delay to the inputs that the PD controller receives ##################
length_of_delay = 0.0001



Time = time.time()
for i, t in enumerate(np.arange(0,int(L),dt)):

    # Update records
    ang_rec[round(t/dt)+1,:]=ang
    vel_rec[round(t/dt)+1,:]=vel
    acc_rec[round(t/dt)+1,:]=acc
    if t>0:
        jerk_rec[round(t/dt)+1,:]=acc-acc_rec[round(t/dt),:]
    
            
    ## Current wrist target
    current_wrist_target=final_wrist_pos[curr_target][:]

    if curr_target<=7:
        ## Planner

        #Get desired position from planner ##########
        if t-start_t<T:
            desired_pos = Sim.minjerk(init_wrist_pos, current_wrist_target, t-start_t)
        
        #Get desired angle from inverse kinematics (Inverse kinematics)
        desired_ang = np.real(Sim.invkinematics(desired_pos))
        # desired_ang[i] = np.array(desired_ang)


        ########### 2.2.0. Initial desired torques from PD controller #############
        # desired_torque = Sim.pdcontroller(desired_ang, ang, vel)
        # desired_torques[i] = np.array(desired_torque)

        ############ Task 2.2.1 Get desired torque from PD controller w. noise (Forward dynamics) #############
        # noise_coefficient = 0.0   #Coefficient of noise variation
        # desired_torque = Sim.pdcontroller(desired_ang, ang, vel, noise_coefficient)  #Adding noise to torque
        # desired_torques[i] = np.array(desired_torque)
        
        ############ Task 2.2.2 Delayed joint angles and velocities #############
        # t: current simulation time
        # [round((t - length_of_delay) / dt): The index of the delayed time step
        # ang_rec: recorded angles 
        # vel_rec: recorded velocities

        delayed_ang = ang_rec[round((t - length_of_delay) / dt), :] # [x,:] all columns of row x are affected
        delayed_vel = vel_rec[round((t - length_of_delay) / dt), :]

        desired_torque = Sim.pdcontroller(desired_ang, delayed_ang, delayed_vel, 0)  #noise = 0
        desired_torques[i] = np.array(desired_torque)

       
        # Pass torque to plant 
        [ang,vel,acc]= Sim.plant(ang,vel,acc,desired_torque)


        # Calculate new joint positions (Forward kinematics)
        [elbow_pos, wrist_pos] = Sim.fkinematics(ang)


        #Record wrist position 
        wrist_pos_rec[round(t/dt)+1] = wrist_pos[0]
 
        #Next target 
        if (t-start_t>=T+.02) & (curr_target<7):
            curr_target=curr_target+1
            init_wrist_pos=wrist_pos
            start_t=t


############## Task 2.2.1 Plot desired torques (coordinates) w. varying noise ##############
# fig1, ax1 = plt.subplots(1, 1, figsize=(12,8))
# ax2 = ax1.twinx()
# ax1.plot(np.arange(int(L/dt)),desired_torques[:,0],color='red')    #Take 1'st column
# ax2.plot(np.arange(int(L/dt)),desired_torques[:,1],color='green')  #Take 2nd column
# ax1.set_ylabel('Torque for joint 1', fontsize=10) 
# ax2.set_ylabel('Torque for joint 2 ', fontsize=10) 
# ax1.set_xlabel('Simulation duration [s]')
# plt.title(r"Desired torque for target joint angels" "\n" r"When arm reaching to 4 targets (1s timestep)")
# plt.show()



############## Task: 2.2.2 Compute torque with delayed angles and velocities ##############
# #Delay is added to the inputs which the pd controller recieves. (noise coefficient is set to 0)
fig1, ax1 = plt.subplots(1, 1, figsize=(12,8))
ax2 = ax1.twinx()
ax1.plot(np.arange(int(L/dt)),desired_torques[:,0],color='red')    #Take 1'st column
ax2.plot(np.arange(int(L/dt)),desired_torques[:,1],color='green')  #Take 2nd column
ax1.set_ylabel('Torque for joint 1', fontsize=10) 
ax2.set_ylabel('Torque for joint 2 ', fontsize=10) 
ax1.set_xlabel('Simulation duration [s]')
plt.title(r"Desired torques w. length_of_delay = 0.0001" "\n" r"noise_coefficient = 0.0" "\n" r" kp = 140, kd = 12 ")
plt.show()



# fig, ax = plt.subplots(1, 1, figsize=(12,8))
# ax.set_xlabel('meters', fontsize=10)
# ax.set_ylabel('meters', fontsize=10)
# ax.set_xlim([-0.5, .5])
# ax.set_ylim([-0.5, .5])




# # Plot arm, wrist path, and targets -- ANIMATION 

#     ax.cla()
#     ax.scatter(np.array(final_wrist_pos)[:,0], np.array(final_wrist_pos)[:,1], color='green')
#     ax1.scatter(np.array(final_wrist_pos)[:,0], np.array(final_wrist_pos)[:,1], color='green')

#     ax.plot([shoulder_pos[0], elbow_pos[0][0]], [shoulder_pos[1], elbow_pos[1][0]], color='blue')
#     ax.plot([elbow_pos[0][0], wrist_pos[0]], [elbow_pos[1][0], wrist_pos[1]], color= 'blue')   
#     plt.pause(0.01)
#     # plt.tight_layout()

#     for t2 in np.arange(dt,t,dt):
#         # ax1.cla()
#         ax.plot(wrist_pos_rec[:round(t2/dt),0], wrist_pos_rec[:round(t2/dt),1],'--',color='red',linewidth=0.5)
        
#         ax.plot([wrist_pos_rec[round(t2/dt),0], wrist_pos_rec[round(t2/dt)+1,0]], [wrist_pos_rec[round(t2/dt),1], wrist_pos_rec[round(t2/dt)+1,1]],color='red',linewidth=0.5)
#         plt.show(block=False)
#         plt.pause(0.01)
#         ax.cla()
#     ax.autoscale_view()

# elapsed = time.time() - Time
# print("Time elapsed:",elapsed)

# ax.plot(wrist_pos_rec[:-1,0], wrist_pos_rec[:-1,1],'--',color='red',linewidth=0.5)
# ax.scatter(np.array(final_wrist_pos)[:,0], np.array(final_wrist_pos)[:,1], color='green')
# plt.show()

# plt.subplot(3,1,1)
# [A,B]= plt.plot(np.arange(0,L-dt,dt), [xx[0] for xx in vel_rec[:int(L/dt)-1]],np.arange(0,L-dt,dt), [xx[1] for xx in vel_rec[:int(L/dt)-1]])
# plt.legend([A,B],['Shoulder','Elbow'])
# plt.xlabel('time (ms)')
# plt.ylabel('velocity')
# plt.subplot(3,1,2)
# [A,B]= plt.plot(np.arange(0,L-dt,dt), [xx[0] for xx in acc_rec[:int(L/dt)-1]],np.arange(0,L-dt,dt), [xx[1] for xx in acc_rec[:int(L/dt)-1]])
# plt.legend([A,B],['Shoulder','Elbow'])
# plt.xlabel('time (ms)')
# plt.ylabel('acceleration')
# plt.subplot(3,1,3)
# [A,B]= plt.plot(np.arange(0,L-dt,dt), [xx[0] for xx in jerk_rec[:int(L/dt)-1]],np.arange(0,L-dt,dt), [xx[1] for xx in jerk_rec[:int(L/dt)-1]])
# plt.legend([A,B],['Shoulder','Elbow'])
# plt.xlabel('time (ms)')
# plt.ylabel('jerk')
# plt.tight_layout()
# plt.show()
