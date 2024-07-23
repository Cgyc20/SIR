import numpy as np
import matplotlib.pyplot as plt
import random

# Load parameters from file
S0, I0, k1nu, k2 = np.loadtxt('parameters.dat')

T = 40  # Total time
t = 0

def ODE(S_init, I_init, dt, start_time,threshold):
    t_vec_ODE = [S_init]
    I_vec_ODE = [I_init]
    t_vec_ODE = [start_time]
    t_new = start_time
    S = S_init
    I = I_init
    while t_new < T and I>=threshold:
        t_new += dt
        S += dt * (-k1nu * S * I)
        I += dt * (k1nu * S * I - k2 * I)

        t_vec_ODE.append(t_new)
        I_vec_ODE.append(I)
        S_vec_ODE.append(S)
    return t_vec_ODE, I_vec_ODE,S_vec_ODE

# Initialize state vector correctly
def propensity(state):
    """This calculates the propensity function"""
    S, I = state  # Extract S and I from the state vector
    a1 = k1nu * S * I  # Reaction 1 propensity
    a2 = k2 * I  # Reaction 2 propensity
    return a1, a2

state = np.array([S0, I0])  # Combined state vector for S and I
I_array = [state[1]]  # Initialize I_array with the initial I value
t_array = [0]

stoich = np.array([(-1, 1), (0, -1)])  # Stoichiometry matrix

# Initialize ODE vectors outside the loop to avoid reference before assignment error
t_vec_ODE = []
I_vec_ODE = []
S_vec_ODE = []
threshold = 60

while t < T:
    """The main loop"""

    if state[1] >= threshold:
        # Switch to ODE
        
        S = state[0]
        I = state[1]
        dt = 0.1
        start_time = t
        t_vec_ODE, I_vec_ODE, S_vec_ODE = ODE(S, I, dt, start_time,threshold)

        break
        
    

    r1 = random.uniform(0, 1)
    r2 = random.uniform(0, 1)

    a1, a2 = propensity(state)
    a0 = a1 + a2

    if a0 > 0:  # Check to prevent division by zero in tau calculation
        tau = np.log(1 / r1) / a0

        if r2 < a1 / a0:
            # Execute reaction 1
            change = stoich[0]  # Select the first reaction's stoichiometry
        else:
            # Execute reaction 2
            change = stoich[1]  # Select the second reaction's stoichiometry

        state += change  # Apply change to the state vector

        state = np.maximum(0, state)  # Ensure state values do not go below 0

        t += tau
        I_array.append(state[1])  # Append the current I value
        t_array.append(t)
    else:
        break  # Break the loop if a0 is not greater than 0 to avoid infinite loop
        

# Plotting
plt.figure()
plt.step(t_array, I_array, label='Stochastic Simulation')
if t_vec_ODE and I_vec_ODE:  # Check if ODE vectors are not empty
    plt.plot(t_vec_ODE, I_vec_ODE, label='ODE Model')
plt.xlabel('Time')
plt.ylabel('Infected Population')
plt.title('Comparison of Infection Spread Models')
plt.legend()
plt.show()