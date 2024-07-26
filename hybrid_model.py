import numpy as np
import matplotlib.pyplot as plt
import random
import tqdm

"""Parameters"""
DS_0 = 400
DI_0 = 5
CS_0 = 0
CI_0 = 0
k1 = 0.002
k2 = 0.1
dt = 0.001
tf = 15
T1 = 200
gamma = 0.1
number_molecules = 4

num_points = int(tf / dt) + 1  # Number of time points in the simulation
timegrid = np.linspace(0, tf, num_points, dtype=np.float64)  # Time points
data_table_init = np.zeros((num_points, number_molecules), dtype=np.float64)  # Matrix to store simulation results
data_table_cum = np.zeros((num_points, number_molecules), dtype=np.float64) #The cumulative simulation results

combined = np.zeros((num_points, 2), dtype=np.float64)

#Note the vector is like this:
"""
Vector = 
[
DS
DI
CS
CI
]
"""
states_init = np.array([DS_0,DI_0,CS_0,CI_0], dtype=float)

S_matrix = np.array([[-1,1,0,0],
                     [0,1,-1,0],
                     [0,-1,0,0],
                     [1,0,-1,0],
                     [0,1,0,-1],
                     [-1,0,1,0],
                     [0,-1,0,1]],dtype=float)

"""FORWARD IS FROM DISCRETE TO CONTINIOUS"""
def compute_propensities(states):

    DS,DI,CS,CI = states
    alpha_1 = k1*DS*DI
    alpha_2 = k1*CS*DI
    alpha_3 = k2*DI

    ### Found a bug (I was taking entire vector sum)
    alpha_bS = gamma * CS if CS+DS <= T1 else 0# Continious S to discrete S

    alpha_bI = gamma * CI if CI+DI  <= T1 else 0 # Cont I to Discrete I

    alpha_fS = gamma * DS if  CS+DS > T1 else 0# Discrete S to continous S
    
    alpha_fI = gamma * DI if CI+DI >T1 else 0 # Discrete I to Cont I
    

    return np.array([alpha_1,alpha_2,alpha_3,alpha_bS,alpha_bI,alpha_fS,alpha_fI])


def perform_reaction(index,states):
    
    DS,DI,CS,CI = states
    """In this case we're interested in the case 
    where Continious to discrete! We don't want contintous to go below 1"""
    #This is the case for reaction 4 and 5 (ie index = 3,4)
    #If index == 3 then we are looking at reaction 4 (Cs --> Ds)
    if index ==3 and CS < 1:

        if CS >= random.uniform(0,1):
            """Then we update Ds"""
            states[0] += S_matrix[index][0]
            states[2] = 0
        else:
            states += S_matrix[index]
          
    elif index == 4 and CI < 1:
        
        if CI >= random.uniform(0,1):
            """Then we update Ds"""
            states[1] += S_matrix[index][1]
            states[3] = 0
        else:
            states += S_matrix[index]
    else:
        """otherwise we just execute the norm """
        states += S_matrix[index]

    return states

def gillespie_step(alpha_cum, alpha0, states):
    """Performs one step of the Gillespie algorithm."""
    r2 = random.uniform(0, 1)  # Generate random number for reaction selection
    index = next(i for i, alpha in enumerate(alpha_cum / alpha0) if r2 <= alpha)  # Determine which reaction occurs
    return perform_reaction(index, states)  # Update states based on selected reaction


def update_ode(states):
    """Updates the states based on the ODE model using forward Euler."""

    """Here we have two differential equations
    DS/Dt = -K_1*S*I
    DI/Dt = -K_1*S*I-K_2*I
    """    
    states[2] = max(states[2] - dt * k1 * states[2] * states[3], 0)  # Update discrete susceptible
    states[3] = max(states[3] - dt *( k1 * states[2]*states[3]-k2*states[3]), 0)  # Update continuous molecules
    return states

t = 0  # Initial time
td = dt  # Initial ODE step time


def run_simulation(num_points):

    """This run the simulation one time
    INPUT: the number of data points
    RETURNS: A data table with the concentrations of D and C respectively"""

    t = 0 #Set time = 0
    old_time = t
    td = dt #Set td to be equal to initial delta t
    data_table = np.zeros((num_points, number_molecules), dtype=np.float64) #Define a new data table
    states = states_init.copy() #

    while t<tf:

        alpha_list = compute_propensities(states) #Compute the propensities
        alpha0 = sum(alpha_list)
        alpha_cum = np.cumsum(alpha_list)

        if alpha0 != 0:
            tau = np.log(1 / random.uniform(0, 1)) / alpha0 

            if t + tau <= td:
                states = gillespie_step(alpha_cum, alpha0, states)
                old_time = t
                t += tau  # Update time

                # Determine indices for updating results
                ind_before = np.searchsorted(timegrid, old_time, 'right')
                ind_after = np.searchsorted(timegrid, t, 'left')

                for index in range(ind_before, min(ind_after + 1, num_points)):
                    data_table[index, :] = states  # Store results in data_table
            else:
                states = update_ode(states)  # Perform ODE step
                t = td
                td += dt

                index = min(np.searchsorted(timegrid, t + 1e-10, 'left'), num_points - 1)
                data_table[index, :] = states  # Store results in data_table
        else:
            states = update_ode(states)  # Perform ODE step
            t = td
            td += dt

            index = min(np.searchsorted(timegrid, t + 1e-10, 'right'), num_points - 1)
            data_table[index, :] = states  # Store results in data_table
    return data_table


total_simulations = 1

for i in tqdm.tqdm(range(total_simulations)):
    data_table_cum += run_simulation(num_points)
# Calculate combined data (total molecules)
#Now we want to divide the elements by the total number of simulations
data_table_cum /= total_simulations 

combined[:,0] = data_table_cum[:,0] + data_table_cum[:,2] #The S
combined[:,1] = data_table_cum[:,1] + data_table_cum[:,3] #The I


threshold = np.ones(num_points)*T1


plt.figure()
plt.plot(timegrid, data_table_cum[:, 1], label='$D_I$')
plt.plot(timegrid, data_table_cum[:, 3], label='$C_I$')
plt.plot(timegrid,combined[:,1],label = '$C_I+D_I$')
plt.legend()
plt.show()


