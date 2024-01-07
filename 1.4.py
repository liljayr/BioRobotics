import numpy as np
import sympy as smp 
import matplotlib.pyplot as plt 
print("Starting: Leaky-integrate and fire function:")
 #T: time variable, how much time we have (0)
 #I: initial current 

# Task 1.1-2 Defining function
def LIF(um_0, I, end_time):          
                                   
    time = 0   #s
    delta_t = 1e-6  #s
    um_t = np.zeros((int(end_time//delta_t))) 
    um_t[0] = um_0  #mV membrane potential

    Rm = 10*1e6     # Resistance (Mega Ohm) 
    Cm = 1*1e-9     # Membrane capacitance (Nano Farad)
    u_rest = -70*1e-3   # mV
    
    u_thresh = -50*1e-3             # Spiking threshold (milli Volt)
    
    for i in range(len(um_t)-1):
        #d_um_t = smp.diff((u_rest - um_t[i] + Rm*I)/(Rm*Cm),time)
        d_um_t = (u_rest - um_t[i] + Rm * I)/(Rm*Cm) * delta_t
        
        um_t[i+1] = um_t[i] + d_um_t

        if um_t[i] > u_thresh: 
            um_t[i+1] = u_rest

        time += delta_t
    
    return um_t   #voltage across the resistor (membrane potential)

#Task 1.3 
def calculate_isi(um_t):
    delta_t = 1e-6      #s
    u_rest = -70*1e-3   # V
    ix = np.where(um_t == u_rest)[0]
    if len(ix) > 1:
        interval = (ix[1]-ix[0])*delta_t
        frequency = 1/interval
    else: 
        interval = 0
        frequency = 0 

    return interval, frequency

#Task 1.4
current = np.arange(1e-10,5e-9,1e-10)
spikeFreq = np.zeros(len(current))
for i, val in enumerate(current): 
    um = LIF(-70e-3,val,0.1)
    spikeFreq[i] = calculate_isi(um)[1]

#1.1
membrane_potential = LIF(-70e-3,2.1e-9,0.1)

#1.2
plt.plot(list(np.arange(0,0.1,1e-6)[:-1]), 1000*membrane_potential)
#plt.plot(membrane_potential)
plt.xlabel("Time [s]")
plt.ylabel("Membrane potential [mV]")
plt.show()

#1.3 Inter-spike interval (ISI) and spiking frequency
data = calculate_isi(membrane_potential)
print(data)
print("Interval: ", data[0]) #0.030445
print("Freq.: ", data[1])    #32.84611594678929

#1.4 
plt.plot(current, spikeFreq)
plt.ylabel("Spike Frequency [Hz]")
plt.xlabel("Current [A]")
plt.show()
