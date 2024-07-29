import numpy as np
import matplotlib.pyplot as plt
import random
import tqdm

S = 400 #Initial discrete Suceptible
I = 1  #Initial discrete Infected
k1 = 0.002 #First rate constant
k2 = 0.1 #Second rate
tf = 50 #Final time
number_molecules = 2 
dt = 0.1


num_points = int(tf / dt) + 1  # Number of time points in the simulation
timegrid = np.linspace(0, tf, num_points, dtype=np.float64)  # Time points
data_table_init = np.zeros((num_points, number_molecules), dtype=np.float64)  # Matrix to store simulation results
data_table_cum = np.zeros((num_points, number_molecules), dtype=np.float64) #The cumulative simulation results

states_init = np.array([S,I], dtype=int) #States vector

#Stoichiometric matrix involved in 
#R1) S + I --> 2I (k1)
#R2) I --> R (k2)
S_matrix = np.array([[-1,1],
                    [0,-1]],dtype=int)


def compute_propensities(states):
    """This will return the 8 propensity functions involved in the 8 different reactions"""
    S,I = states
    """First are the propensities involved in the dynamics"""
    alpha_1 = k1*S*I
    alpha_2 = k2*I
    return np.array([alpha_1,alpha_2],dtype=float)


def perform_reaction(index,states):
    """Here we perform the reaction step. This updates the state vector of the corresponding reaction that occurs
    The reaction will occurs is given by the index calculated in the stochastic loop"""

    """In this case we're interested in the case 
    where Continious to discrete! We don't want contintous to go below 1"""
    #This is the case for reaction 5 and 6 (ie index = 4,5)
    #If index == 4 then we are looking at reaction 5 (Cs --> Ds)
    states += S_matrix[index]  # General update for other reactions
    states[0] = np.max(states[0],0)
    states[1] = np.max(states[1],0)

    return states

def gillespie_step(alpha_cum, alpha0, states):
    """Performs one step of the Gillespie algorithm."""
    r2 = random.uniform(0, 1)  # Generate random number for reaction selection
    index = next(i for i, alpha in enumerate(alpha_cum / alpha0) if r2 <= alpha)  # Determine which reaction occurs
    return perform_reaction(index, states)  # Update states based on selected reaction

def run_simulation(num_points):

    """This run the simulation one time
    INPUT: the number of data points
    RETURNS: A data table with the concentrations of D and C respectively"""

    t = 0 #Set time = 0
    old_time = t
    data_table = np.zeros((num_points, number_molecules), dtype=np.float64) #Define a new data table
    states = states_init.copy() #

    while t<tf:

        
        alpha_list = compute_propensities(states) #Compute the propensities
        alpha0 = sum(alpha_list) #The sum of the list
        if alpha0 == 0:
            break
        alpha_cum = np.cumsum(alpha_list) #Cumulative list. used to find the index 
        #print(f"Cumulative alpha_list = {alpha_cum} ")
        tau = np.log(1 / random.uniform(0, 1)) / alpha0  #Calculate the time value Tau
        #print(f"tau value {tau}")
        states = gillespie_step(alpha_cum, alpha0, states) #Update the states via Gillespie
        old_time = t  #Old time
        t += tau  # Update time

        # Determine indices for updating results
        ind_before = np.searchsorted(timegrid, old_time, 'right')
        ind_after = np.searchsorted(timegrid, t, 'left')
        # print(f"old_time: {old_time}, t: {t}, ind_before: {ind_before}, ind_after: {ind_after}")
        for index in range(ind_before, min(ind_after + 1, num_points)):
            data_table[index, :] = states  # Store results in data_table    

    return data_table

total_simulations = 100 #Total sim

for i in tqdm.tqdm(range(total_simulations)): #WE run all simulations, compile in a total list and find average 
    data_table_cum += run_simulation(num_points)
# Calculate combined data (total molecules)
#Now we want to divide the elements by the total number of simulations
data_table_cum /= total_simulations 

plt.figure()

# Plot for Susceptible (S)
plt.subplot(2, 1, 1)
plt.plot(timegrid, data_table_cum[:, 0], label='S')
plt.xlabel('Time')
plt.ylabel('Susceptible')
plt.legend()

# Plot for Infected (I)
plt.subplot(2, 1, 2)
plt.plot(timegrid, data_table_cum[:, 1], label='I')
plt.xlabel('Time')
plt.ylabel('Infected')
plt.legend()

plt.tight_layout()
plt.show()



