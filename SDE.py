import numpy as np
import matplotlib.pyplot as plt
import random

# Load parameters from file
S0, I0, k1nu, k2 = np.loadtxt('parameters.dat')
# Total time
T = 40


def ODE(S_value, I_value, t_value, dt):

    """This models the average trajectory according to the CME
    INPUTS :: The initial S0,I0 and initial timestep t_value, timestep dt
    OUTPUT :: List of I,S,T (which is the vector list)"""
    t_list = [t_value]
    S_list = [S_value]
    I_list = [I_value]
    

    while t_value < T :
    
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

def gillespie(S_value, I_value, t_value):
    """This uses the Gillespie algorithm to model the system """

    I = I_value
    S = S_value
    t = t_value
    I_list = [I_value]
    S_list = [S_value]
    t_list = [t_value]

    #Can add another condition to only run up to this threshold value
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

# Perform Gillespie simulation

simulations = 10
I_all_list = []
t_all_list = []
for i in range(simulations):
    """Plotting range of trajectories"""
    _, I_list, t_list = gillespie(S0, I0, t_value)
    I_all_list.append(I_list)
    t_all_list.append(t_list)

S_list, I_list, t_list = ODE(S0, I0, t_value, 0.1)



# Plot the results

plt.figure(figsize=(8, 5))  # Increase the height of the figure
for i in range(simulations):
    plt.step(t_all_list[i], I_all_list[i], where='post', linewidth=0.5)  # SDEs with thinner lines
plt.plot(t_list, I_list, linewidth=2,linestyle='--',color = 'black')  # ODE with dotted and thicker line
plt.xlabel('Time')
plt.ylabel('Infected Population')
plt.grid()
plt.legend(['SDE', 'ODE'], loc='best')  # Add legend to distinguish between SDE and ODE
plt.show()
