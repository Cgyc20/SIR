import numpy as np
import matplotlib.pyplot as plt

# Parameters
k1 = 0.002 #First rate constant
k2 = 0.1 #Second rate
dt = 0.1  # Time step
tf = 40  # Final time
S0 = 400  # Initial proportion of susceptible individuals
I0 = 1  # Initial proportion of infected individuals


num_points = int(tf / dt) + 1
timegrid = np.linspace(0, tf, num_points)

# Arrays to store the results
S = np.zeros(num_points)
I = np.zeros(num_points)
R = np.zeros(num_points)

# Initial conditions
S[0] = S0
I[0] = I0

def differential(S, I):
    dSdt = -k1 * S * I
    dIdt = k1 * S * I - k2 * I
    return dSdt, dIdt

def RK4_step(S, I,  dt):
    k1_S, k1_I = differential(S, I)
    k2_S, k2_I = differential(S + k1_S * dt / 2, I + k1_I * dt / 2)
    k3_S, k3_I = differential(S + k2_S * dt / 2, I + k2_I * dt / 2)
    k4_S, k4_I = differential(S + k3_S * dt, I + k3_I * dt)
    S_next = S + dt * (k1_S + 2 * k2_S + 2 * k3_S + k4_S) / 6
    I_next = I + dt * (k1_I + 2 * k2_I + 2 * k3_I + k4_I) / 6
    return S_next, I_next

for i in range(1, num_points):
    S[i], I[i] = RK4_step(S[i-1], I[i-1], dt)

plt.plot(timegrid, S, label='Susceptible')
plt.plot(timegrid, I, label='Infected')
plt.xlabel('Time')
plt.ylabel('Proportion')
plt.legend()
plt.show()







