import numpy as np
import matplotlib.pyplot as plt

# Parameters
k1 = 0.002  # First rate constant
k2 = 0.1  # Second rate constant
dt = 0.1  # Time step
tf = 40  # Final time
S0 = 400  # Initial number of susceptible individuals
I0 = 1  # Initial number of infected individuals

num_points = int(tf / dt) + 1
timegrid = np.linspace(0, tf, num_points)

# Arrays to store the results
S = np.zeros(num_points)
I = np.zeros(num_points)
SI = np.zeros(num_points)
S2 = np.zeros(num_points)
I2 = np.zeros(num_points)

# Initial conditions
S[0] = S0
I[0] = I0
SI[0] = S0 * I0
S2[0] = S0 ** 2
I2[0] = I0 ** 2

epsilon = 1e-10  # Small value to avoid division by zero

def differential(S, I, SI, S2, I2):
    DSIDT = k1 * ((S2 * SI ** 2) / (I * S ** 2 + epsilon) - ((SI ** 2) * I2) / (S * I ** 2 + epsilon) - SI) - k2 * SI
    DS2DT = k1 * (SI - 2 * ((SI ** 2) * S2) / (S ** 2 * I + epsilon))
    DI2DT = k1 * (2 * ((SI ** 2) * I2) / (S * I ** 2 + epsilon) + SI) + k2 * (I - 2 * I2)

    dSdt = -k1 * SI
    dIdt = k1 * SI - k2 * I

    return dSdt, dIdt, DSIDT, DS2DT, DI2DT

def RK4(S, I, SI, S2, I2):
    k1_S, k1_I, k1_SI, k1_S2, k1_I2 = differential(S, I, SI, S2, I2)
    k2_S, k2_I, k2_SI, k2_S2, k2_I2 = differential(S + k1_S * dt / 2, I + k1_I * dt / 2, SI + k1_SI * dt / 2, S2 + k1_S2 * dt / 2, I2 + k1_I2 * dt / 2)
    k3_S, k3_I, k3_SI, k3_S2, k3_I2 = differential(S + k2_S * dt / 2, I + k2_I * dt / 2, SI + k2_SI * dt / 2, S2 + k2_S2 * dt / 2, I2 + k2_I2 * dt / 2)
    k4_S, k4_I, k4_SI, k4_S2, k4_I2 = differential(S + k3_S * dt, I + k3_I * dt, SI + k3_SI * dt, S2 + k3_S2 * dt, I2 + k3_I2 * dt)

    S_next = S + dt * (k1_S + 2 * k2_S + 2 * k3_S + k4_S) / 6
    I_next = I + dt * (k1_I + 2 * k2_I + 2 * k3_I + k4_I) / 6
    SI_next = SI + dt * (k1_SI + 2 * k2_SI + 2 * k3_SI + k4_SI) / 6
    S2_next = S2 + dt * (k1_S2 + 2 * k2_S2 + 2 * k3_S2 + k4_S2) / 6
    I2_next = I2 + dt * (k1_I2 + 2 * k2_I2 + 2 * k3_I2 + k4_I2) / 6

    return S_next, I_next, SI_next, S2_next, I2_next

for i in range(1, num_points):
    S[i], I[i], SI[i], S2[i], I2[i] = RK4(S[i-1], I[i-1], SI[i-1], S2[i-1], I2[i-1])

plt.plot(timegrid, S, label='Susceptible')
plt.plot(timegrid, I, label='Infected')
plt.xlabel('Time')
plt.ylabel('Proportion')
plt.legend()
plt.show()
