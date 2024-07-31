import numpy as np
import matplotlib.pyplot as plt
from Modelling import HybridModel, SISimulation
import tqdm



# Parameters for the simulation

DI_0 = 2        # Initial number of discrete Infected individuals
CS_0 = 0        # Initial number of continuous Susceptible individuals
CI_0 = 0        # Initial number of continuous Infected individuals
k1 = 0.002      # Rate constant for infection
k2 = 0.1        # Rate constant for recovery
dt = 0.2        # Time step for ODE (Ordinary Differential Equations)
tf = 100         # Final time for the simulation
T1 = 40         # Threshold for converting continuous to discrete Infected
T2 = T1         # Threshold for converting continuous to discrete Susceptible
gamma = 2    # Rate of conversion between discrete and continuous populations
total_sims = 100  # Number of simulations to run

discrete_susceptible_vector = np.arange(1000,3000,200)

time_array = np.zeros((len(discrete_susceptible_vector),3),dtype = np.float64)


for i in range(len(discrete_susceptible_vector)):
    susceptible_number = discrete_susceptible_vector[i]
    print(susceptible_number)
    DS_0 = susceptible_number      # Initial number of discrete Susceptible individuals

    combined_model = SISimulation(S0=DS_0, I0=DI_0, k1=k1, k2=k2, tf=tf, dt=dt)
    # Run the combined model using SISimulation to get the ODE and SSA results
    S_ODE, I_ODE, S_stochastic, I_stochastic = combined_model.run_combined(total_simulations=total_sims)

    T1 = int(max(I_stochastic)/10)
    T2 = T1

    # Create an instance of the HybridModel with the specified parameters
    hybrid_model = HybridModel(
        DS_0=DS_0, DI_0=DI_0, CS_0=CS_0, CI_0=CI_0,
        k1=k1, k2=k2, dt=dt, tf=tf, T1=T1, T2=T2, gamma=gamma
    )

    # Create an instance of the SISimulation with the specified parameters

    # Run multiple simulations using HybridModel to get the average results
    timegrid, DS_vector, DI_vector, CS_vector, CI_vector, HS_vector, HI_vector = hybrid_model.run_multiple(total_simulations=total_sims)

    SSA_time = combined_model.total_SSA_time
    ODE_time = combined_model.total_ODE_time
    hybrid_time = hybrid_model.total_time

    time_array[i,0] = SSA_time
    time_array[i,1] = ODE_time
    time_array[i,2] = hybrid_time

plt.figure()

plt.plot(discrete_susceptible_vector,time_array[:,0],label = 'SSA')
plt.plot(discrete_susceptible_vector,time_array[:,1],label = 'ODE')
plt.plot(discrete_susceptible_vector,time_array[:,2],label = 'Hybrid')
plt.xlabel('Number of discrete susceptible individuals')
plt.ylabel('Time taken (s)')
plt.legend()
plt.show()
 
