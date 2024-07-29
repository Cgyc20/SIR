import numpy as np
import matplotlib.pyplot as plt
import random
import tqdm

"""Parameters"""
DS_0 = 400 #Initial discrete Suceptible
DI_0 = 1  #Initial discrete Infected
CS_0 = 0 #Initial continious Suceptible
CI_0 = 0 #Initial continious Infected
k1 = 0.002 #First rate constant
k2 = 0.1 #Second rate
dt = 0.1 #Time step (For ODE)
tf = 30 #Final time
T1 = 50 #Threshold for conversion (Infected)
T2 = 50 #Threshold for conversion (suceptible)
gamma = 0.1 #The rate of conversion 
number_molecules = 4 #The total molecules (two discrete,two cont)

num_points = int(tf / dt) + 1  # Number of time points in the simulation
timegrid = np.linspace(0, tf, num_points, dtype=np.float64)  # Time points
data_table_init = np.zeros((num_points, number_molecules), dtype=np.float64)  # Matrix to store simulation results
data_table_cum = np.zeros((num_points, number_molecules), dtype=np.float64) #The cumulative simulation results

combined = np.zeros((num_points, 2), dtype=np.float64) #Combined totals

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
states_init = np.array([DS_0,DI_0,CS_0,CI_0], dtype=float) #States vector

"""The stoichiometric matrix correpsonding to reactions 1-8"""
S_matrix = np.array([[-1,1,0,0],
                     [0,1,-1,0],
                     [0,-1,0,0],
                     [-1,0,0,1],
                     [1,0,-1,0],
                     [0,1,0,-1],
                     [-1,0,1,0],
                     [0,-1,0,1]],dtype=int)

#Briefly remove that 4th column
# S_matrix = np.array([[-1,1,0,0],
#                      [0,1,-1,0],
#                      [0,-1,0,0],
#                      [1,0,-1,0],
#                      [0,1,0,-1],
#                      [-1,0,1,0],
#                      [0,-1,0,1]],dtype=int)


"""FORWARD IS FROM DISCRETE TO CONTINIOUS"""
def compute_propensities(states):
    """This will return the 8 propensity functions involved in the 8 different reactions"""
    DS,DI,CS,CI = states
    """First are the propensities involved in the dynamics"""
    alpha_1 = k1*DS*DI
    alpha_2 = k1*CS*DI
    alpha_3 = k2*DI
    alpha_4 = k1*DS*CI
    

    """The next are the propensities involved in conversion"""
    ### Found a bug (I was taking entire vector sum)
    alpha_bS = gamma * CS if CS+DS < T2 else 0# Continious S to discrete S

    alpha_bI = gamma * CI if CI+DI  < T1 else 0 # Cont I to Discrete I

    alpha_fS = gamma * DS if  CS+DS >= T2 else 0# Discrete S to continous S
    
    alpha_fI = gamma * DI if CI+DI >= T1 else 0 # Discrete I to Cont I
    
    #Return this as an array
    # return np.array([alpha_1,alpha_2,alpha_3,alpha_bS,alpha_bI,alpha_fS,alpha_fI])
    return np.array([alpha_1,alpha_2,alpha_3,alpha_4,alpha_bS,alpha_bI,alpha_fS,alpha_fI])


def perform_reaction(index,states):
    """Here we perform the reaction step. This updates the state vector of the corresponding reaction that occurs
    The reaction will occurs is given by the index calculated in the stochastic loop"""

    _,_,CS,CI = states
    """In this case we're interested in the case 
    where Continious to discrete! We don't want contintous to go below 1"""
    #This is the case for reaction 5 and 6 (ie index = 4,5)
    #If index == 4 then we are looking at reaction 5 (Cs --> Ds)
    if index == 4 and CS < 1:
        if CS >= random.uniform(0, 1):
            states[0] += S_matrix[index][0]  # Update discrete molecules
            states[2] = 0  # Reset continuous molecules
    elif index == 5 and CI < 1:
        if CI >= random.uniform(0, 1):
            states[1] += S_matrix[index][1]  # Update discrete molecules
            states[3] = 0  # Reset continuous molecules
    else:
        states += S_matrix[index]  # General update for other reactions

    return states

def gillespie_step(alpha_cum, alpha0, states):
    """Performs one step of the Gillespie algorithm."""
    r2 = random.uniform(0, 1)  # Generate random number for reaction selection
    index = next(i for i, alpha in enumerate(alpha_cum / alpha0) if r2 <= alpha)  # Determine which reaction occurs
    return perform_reaction(index, states)  # Update states based on selected reaction


def update_ode(states):
    """Updates the states based on the ODE model using forward Euler. We use RK4 to calculate this"""

    """Here we have two differential equations
    DS/Dt = -K_1*S*I
    DI/Dt = -K_1*S*I-K_2*I
    """    

    def differential(S,I): 
        """Note these must be continious version of particles"""
        DsDt = -k1*S*I
        DiDt = k1*S*I-k2*I
        return DsDt, DiDt
        """Differential calculation"""

    def RK4(states):
        _,_,S,I = states
        """RK4 Approximation of next time step"""
        P1 = differential(S,I)
        P2 = differential(S+P1[0]*dt/2,I+P1[1]*dt/2)
        P3 = differential(S+P2[0]*dt/2,I+P2[1]*dt/2)
        P4 = differential(S+P3[0]*dt,I+P3[1]*dt)
        return S + (P1[0]+2*P2[0]+2*P3[0]+P4[0])*dt/6, I + (P1[1]+2*P2[1]+2*P3[1]+P4[1])*dt/6
    
    #Now return these states again. Note we cannot have negative states
    rk4_result = RK4(states)
    states[2] = max(rk4_result[0], 0)
    states[3] = max(rk4_result[1], 0)

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
        alpha0 = sum(alpha_list) #The sum of the list
        alpha_cum = np.cumsum(alpha_list) #Cumulative list. used to find the index 

        if alpha0 != 0: #If alpha0 is not 0 then we can do the SSA

            tau = np.log(1 / random.uniform(0, 1)) / alpha0  #Calculate the time value Tau

            if t + tau <= td: #If t+tau is less than the NEXT ODE timestep then we perform the SSA

                states = gillespie_step(alpha_cum, alpha0, states) #Update the states via Gillespie
                old_time = t  #Old time
                t += tau  # Update time

                # Determine indices for updating results
                ind_before = np.searchsorted(timegrid, old_time, 'right')
                ind_after = np.searchsorted(timegrid, t, 'left')
                # print(f"old_time: {old_time}, t: {t}, ind_before: {ind_before}, ind_after: {ind_after}")
                for index in range(ind_before, min(ind_after + 1, num_points)):
                    data_table[index, :] = states  # Store results in data_table
            else:
                states = update_ode(states)  # Perform ODE step
                t = td
                td += dt

                index = min(np.searchsorted(timegrid, t + 1e-10, 'left'), num_points - 1)
                data_table[index, :] = states  # Store results in data_table
        else:
            states = update_ode(states)  # Perform ODE step! 
            t = td
            td += dt

            index = min(np.searchsorted(timegrid, t + 1e-10, 'right'), num_points - 1)
            data_table[index, :] = states  # Store results in data_table

    return data_table


total_simulations = 5 #Total sim

for i in tqdm.tqdm(range(total_simulations)): #WE run all simulations, compile in a total list and find average 
    data_table_cum += run_simulation(num_points)
# Calculate combined data (total molecules)
#Now we want to divide the elements by the total number of simulations
data_table_cum /= total_simulations 

combined[:,0] = data_table_cum[:,0] + data_table_cum[:,2] # The S
combined[:,1] = data_table_cum[:,1] + data_table_cum[:,3] # The I

threshold = np.ones(num_points) * T1 # The threshold line 

"""Plotting the results"""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# Plot current infected on the first subplot
ax1.plot(timegrid, data_table_cum[:, 1], label='$D_I$ Discrete')
ax1.plot(timegrid, data_table_cum[:, 3], label='$C_I$ Continuous')
ax1.plot(timegrid, combined[:, 1], label='$C_I+D_I$ Combined', color='black', linestyle='--')
ax1.plot(timegrid, threshold, '--', label='Conversion Threshold')
ax1.set_xlabel('days')
ax1.set_ylabel('Number infected')
ax1.legend()
ax1.grid(True)
ax1.set_title('Current Infected Over Time')

# Plot susceptible on the second subplot
ax2.plot(timegrid, data_table_cum[:, 0], label='$D_S$ Discrete')
ax2.plot(timegrid, data_table_cum[:, 2], label='$C_S$ Continuous')
ax2.plot(timegrid, combined[:, 0], label='$C_S+D_S$ Combined', color='black', linestyle='--')
ax2.set_xlabel('days')
ax2.set_ylabel('Number susceptible')
ax2.legend()
ax2.grid(True)
ax2.set_title('Susceptible Over Time')

plt.tight_layout()
plt.show()


