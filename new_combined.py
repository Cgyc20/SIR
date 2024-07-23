import numpy as np
import matplotlib.pyplot as plt
import random

# Load parameters from file
S0, I0, k1nu, k2 = np.loadtxt('parameters.dat')

# Total time
T = 40

def ODE(S_value, I_value, t_value, dt, threshold):
    t_list = [t_value]
    S_list = [S_value]
    I_list = [I_value]
    


    while t_value < T and I_value >= threshold-1:
        # if I_value <= threshold+1:
        #     break  # Stop the simulation if the threshold is reached
    
        t_value += dt
        dS = -k1nu * S_value * I_value * dt
        dI = (k1nu * S_value * I_value - k2 * I_value) * dt

        S_value += dS
        I_value += dI

        t_list.append(t_value)
        I_list.append(I_value)
        S_list.append(S_value)
        
        if I_value <= 0:
            break  # Stop if the infected population drops to 0 or below

    return S_list, I_list, t_list

def propensity(S, I):
    """This calculates the propensity function"""
    a1 = k1nu * S * I  # Reaction 1 propensity
    a2 = k2 * I  # Reaction 2 propensity
    return a1, a2

stoich = np.array([(-1, 1), (0, -1)])  # Stoichiometry matrix

def gillespie(S_value, I_value, t_value, threshold):
    I = I_value
    S = S_value
    t = t_value
    I_list = [I_value]
    S_list = [S_value]
    t_list = [t_value]

   

    while t < T:

        r1 = random.uniform(0, 1)
        r2 = random.uniform(0, 1)

        a1, a2 = propensity(S, I)
        a0 = a1 + a2

        if a0 > 0:  # Check to prevent division by zero in tau calculation
            tau = np.log(1 / r1) / a0

            if r2 < a1 / a0:
                # Execute reaction 1
                change = stoich[0]  # Select the first reaction's stoichiometry
            else:
                # Execute reaction 2
                change = stoich[1]  # Select the second reaction's stoichiometry

            S += change[0]
            I += change[1]

            S = max(0, S)  # Ensure S does not go below 0
            I = max(0, I)  # Ensure I does not go below 0

            t += tau
            I_list.append(I)  # Append the current I value
            S_list.append(S)
            t_list.append(t)
        
    return S_list, I_list, t_list

# Initialize lists
S_list_1 = [S0]
I_list_1 = [I0]
t_value = 0
t_list_1 = [t_value]

# Simulation parameters
dt = 0.1
threshold = 80

# Perform Gillespie simulation


S_list_1, I_list_1, t_list_1 = gillespie(S0, I0, t_value, threshold)

for i, elements in enumerate(I_list_1):
    if elements >= threshold:
        index_value = i
        break
print(index_value)

S_list_restricted = S_list_1[:index_value]
I_list_restricted = I_list_1[:index_value]
t_list_restricted = t_list_1[:index_value]


S0 = S_list_restricted[-1]
I0 = I_list_restricted[-1]
t_value = t_list_restricted[-1]
print(I0)

length_orig = len(S_list_1)
# Perform ODE simulation starting from the end of the Gillespie simulation
S_list_new, I_list_1_new, t_list_1_new = ODE(S0, I0, t_value, dt, threshold)

S_list_2, I_list_2, t_list_2 = gillespie(S_list_new[-1], I_list_1_new[-1], t_list_1_new[-1], threshold)

# Plot the results
plt.figure()
plt.step(t_list_1,I_list_1, where='post', label='Stochastic Simulation')
plt.plot(t_list_1_new,I_list_1_new,label='ODE')
plt.step(t_list_2,I_list_2, where='post', label='Stochastic Simulation')

plt.xlabel('Time')
plt.ylabel('Infected Population')
plt.legend()
plt.show()
