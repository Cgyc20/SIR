import numpy as np
import matplotlib.pyplot as plt
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
    
from Modelling import HybridModel, SISimulation #type: ignore 
import tqdm

# Parameters for the simulation
DS_0 = 200
DI_0 = 1       # Initial number of discrete Infected individuals
CS_0 = 0        # Initial number of continuous Susceptible individuals
CI_0 = 0        # Initial number of continuous Infected individuals
k1 = 0.002      # Rate constant for infection
k2 = 0.1        # Rate constant for recovery
dt = 0.2        # Time step for ODE (Ordinary Differential Equations)
tf = 60         # Final time for the simulation

gamma = 2    # Rate of conversion between discrete and continuous populations
total_sims = 50  # Number of simulations to run

T1 = 50        # Threshold for converting continuous to discrete Infected
T2 = 52         # Threshold for converting continuous to discrete Susceptible

fixed_threshold_value = 20

below_threshold = fixed_threshold_value -10
above_threshold = fixed_threshold_value  

total_repeats = 1000

hybrid_model_fixed_threshold = HybridModel(
    DS_0=DS_0, DI_0=DI_0, CS_0=CS_0, CI_0=CI_0,
    k1=k1, k2=k2, dt=dt, tf=tf, T1=fixed_threshold_value, T2=fixed_threshold_value, gamma=gamma
)
hybrid_model_different_threshold = HybridModel(
    DS_0=DS_0, DI_0=DI_0, CS_0=CS_0, CI_0=CI_0,
    k1=k1, k2=k2, dt=dt, tf=tf, T1=below_threshold, T2=above_threshold, gamma=gamma
)

fixed_threshold_time_vector = np.zeros(total_repeats, dtype=np.float64)
altered_threshold_time_vector = np.zeros(total_repeats, dtype=np.float64)

# Run simulations for fixed threshold
for i in range(total_repeats):
    print(f"Fixed Threshold Simulation {i + 1}/{total_repeats}")
    _, _, _, _, _, _, _ = hybrid_model_fixed_threshold.run_multiple(total_simulations=total_sims)
    fixed_threshold_time = hybrid_model_fixed_threshold.total_time
    fixed_threshold_time_vector[i] = fixed_threshold_time

# Run simulations for altered threshold
for i in range(total_repeats):
    print(f"Altered Threshold Simulation {i + 1}/{total_repeats}")
    _, _, _, _, _, _, _ = hybrid_model_different_threshold.run_multiple(total_simulations=total_sims)
    altered_threshold_time = hybrid_model_different_threshold.total_time
    altered_threshold_time_vector[i] = altered_threshold_time

# Plotting the histograms
plt.figure(figsize=(12, 8))

plt.hist(fixed_threshold_time_vector, bins=40, alpha=0.5, density=True, label='Fixed Threshold')
plt.hist(altered_threshold_time_vector, bins=40, alpha=0.5, density=True, label='Altered Threshold')

plt.xlabel('Time taken (s)', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.title('Comparison of Efficiency with Different Thresholds', fontsize=16)
plt.legend()
plt.grid(True)
plt.show()
