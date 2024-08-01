import numpy as np
import matplotlib.pyplot as plt
from Modelling import HybridModel, SISimulation
import tqdm


# Parameters for the simulation

DS_0 = 200
DI_0 = 1       # Initial number of discrete Infected individuals
CS_0 = 0        # Initial number of continuous Susceptible individuals
CI_0 = 0        # Initial number of continuous Infected individuals
k1 = 0.002      # Rate cons tant for infection
k2 = 0.1        # Rate constant for recovery
dt = 0.2        # Time step for ODE (Ordinary Differential Equations)
tf = 60         # Final time for the simulation

gamma = 2    # Rate of conversion between discrete and continuous populations
total_sims = 10  # Number of simulations to run

T1 = 20         # Threshold for converting continuous to discrete Infected
T2 = 25         # Threshold for converting continuous to discrete Susceptible


total_repeats = 100

hybrid_model_fixed_threshold = HybridModel(
    DS_0=DS_0, DI_0=DI_0, CS_0=CS_0, CI_0=CI_0,
    k1=k1, k2=k2, dt=dt, tf=tf, T1=T1, T2=T1, gamma=gamma
)
hybrid_model_different_threshold = HybridModel(
    DS_0=DS_0, DI_0=DI_0, CS_0=CS_0, CI_0=CI_0,
    k1=k1, k2=k2, dt=dt, tf=tf, T1=T1, T2=T2, gamma=gamma
)


fixed_threshold_time_vector = np.zeros(total_repeats, dtype = np.float64)
altered_threshold_time_vector = np.zeros(total_repeats, dtype = np.float64)

for i in range(total_repeats):
    print(i)
  
    """
    First we run over N total repeats to work out the time for each simulation 
    For the fixed threshold
    """
    _, _, _, _, _, _, _ = hybrid_model_fixed_threshold.run_multiple(total_simulations=total_sims)
    fixed_threshold_time = hybrid_model_fixed_threshold.total_time
    fixed_threshold_time_vector[i] = fixed_threshold_time
    

for i in range(total_repeats):
    """
    We then do the same for the different threshold! We do in seperate loop, in case theres an issue with computational efficiency
    """
    print(i)
    _, _, _, _, _, _, _ = hybrid_model_different_threshold.run_multiple(total_simulations=total_sims)
    altered_threshold_time = hybrid_model_different_threshold.total_time
    altered_threshold_time_vector[i] = altered_threshold_time



plt.figure(figsize=(10, 6))

plt.hist(fixed_threshold_time_vector, bins=50, alpha=0.5, label='Fixed Threshold')
plt.hist(altered_threshold_time_vector, bins=50, alpha=0.5, label='Altered Threshold')

plt.xlabel('Time taken (s)')
plt.ylabel('Frequency')
plt.title('Comparison of efficiency with different thresholds')
plt.legend()
plt.show()


