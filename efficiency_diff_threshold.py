import numpy as np
import matplotlib.pyplot as plt
from Modelling import HybridModel, SISimulation
import tqdm


# Parameters for the simulation

DI_0 = 2        # Initial number of discrete Infected individuals
CS_0 = 0        # Initial number of continuous Susceptible individuals
CI_0 = 0        # Initial number of continuous Infected individuals
k1 = 0.002      # Rate cons tant for infection
k2 = 0.1        # Rate constant for recovery
dt = 0.2        # Time step for ODE (Ordinary Differential Equations)
tf = 60         # Final time for the simulation

gamma = 2    # Rate of conversion between discrete and continuous populations
total_sims = 1000  # Number of simulations to run


T1 = 15         # Threshold for converting continuous to discrete Infected
T2 = 25         # Threshold for converting continuous to discrete Susceptible

discrete_susceptible_vector = np.arange(150,600,50)

time_array = np.zeros((len(discrete_susceptible_vector),2),dtype = np.float64)


for i in range(len(discrete_susceptible_vector)):
    susceptible_number = discrete_susceptible_vector[i]
    print(f"iteration: {susceptible_number}")
    DS_0 = susceptible_number      # Initial number of discrete Susceptible individuals

    # combined_model = SISimulation(S0=DS_0, I0=DI_0, k1=k1, k2=k2, tf=tf, dt=dt)
    # # Run the combined model using SISimulation to get the ODE and SSA results
    # S_ODE, I_ODE, S_stochastic, I_stochastic = combined_model.run_combined(total_simulations=total_sims)

    #T1 = int(max(I_stochastic)/10)

    # Create an instance of the HybridModel with the specified parameters
    hybrid_model_fixed_threshold = HybridModel(
        DS_0=DS_0, DI_0=DI_0, CS_0=CS_0, CI_0=CI_0,
        k1=k1, k2=k2, dt=dt, tf=tf, T1=T1, T2=T1, gamma=gamma
    )

    hybrid_model_different_threshold = HybridModel(
        DS_0=DS_0, DI_0=DI_0, CS_0=CS_0, CI_0=CI_0,
        k1=k1, k2=k2, dt=dt, tf=tf, T1=T1, T2=T2, gamma=gamma
    )



    # Create an instance of the SISimulation with the specified parameters

    # Run multiple simulations using HybridModel to get the average results
    
    timegrid, DS_vector, DI_vector, CS_vector, CI_vector, HS_vector, HI_vector = hybrid_model_fixed_threshold.run_multiple(total_simulations=total_sims)
    _, _, _, _, _, _, _ = hybrid_model_different_threshold.run_multiple(total_simulations=total_sims)

    fixed_threshold_time = hybrid_model_fixed_threshold.total_time
    altered_threshold_time = hybrid_model_different_threshold.total_time

    time_array[i,0] = fixed_threshold_time
    time_array[i,1] = altered_threshold_time


plt.figure(figsize=(10, 6))

# Define colors and line styles
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Colors suitable for colorblind viewers
linestyles = ['-', '--', ':']  # Different line styles

plt.plot(discrete_susceptible_vector, time_array[:,0], label='Fixed Threshold', color=colors[0], linestyle=linestyles[0])

plt.plot(discrete_susceptible_vector, time_array[:,1], label='Altered Threshold', color=colors[1], linestyle=linestyles[1])

plt.xlabel('Initial number of discrete Susceptible individuals')
plt.ylabel('Time taken (s)')
plt.legend()
plt.show()


