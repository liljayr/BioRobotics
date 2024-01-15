import numpy as np
import matplotlib.pyplot as plt


#Simple feedback controller w 1. order plant (CLOSED LOOPED SYSTEM)
# output y = 1 (adjusted according to gain K for 30 time steps 
# by the difference between its current and target value)
# target = 0

# K: The controller gain (The factor which the input signal is multiplied with to produce the output)



## Initialization
simlen = 30 #simulationlength (time steps)
y = np.zeros((simlen,4)) #output
target = 0.0
K = 1.8 #controller gain
y[0] = 1



for delay in range(4):
    
    y[0,delay] = 1  #[timestep, delayed values], y = 1 is the initial output of the system (when delay = current loop value )


    for t in range(simlen-1): #[0:28]
        delayed_output = y[max(0,t-delay)] #in case of t-delay being negative
        u = K * (target - delayed_output)
        y[t+1]=(0.5 * y[t] + 0.4*u) # 1st order dynamics


## Plot the output y over all time steps 
time = range(simlen)
plt.plot(time, y)

for delay in range(4):
    plt.plot(time, y[:, delay], label=f'Delay={delay}')

plt.axhline(y=target, color='r', linestyle='--', label='Target')
plt.title("K = 1.8")
plt.xlabel('time step')
plt.ylabel('y')
plt.show()

