import numpy as np
import matplotlib.pyplot as plt
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
    
from Modelling import HybridModel, HybridModelLogistic #type: ignore 

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
total_sims = 20  # Number of simulations to run

T1 = 50        # Threshold for converting continuous to discrete Infected
T2 = T1        # Threshold for converting continuous to discrete Susceptible

total_repeats = 400

hybrid_model= HybridModel(
    DS_0=DS_0, DI_0=DI_0, CS_0=CS_0, CI_0=CI_0,
    k1=k1, k2=k2, dt=dt, tf=tf, T1=T1, T2=T2, gamma=gamma
)

hybrid_model_logistic= HybridModelLogistic(DS_0=DS_0, DI_0=DI_0, CS_0=CS_0, CI_0=CI_0, k1=k1, k2=k2, dt=dt, tf=tf, threshold_centre_infected = T1, threshold_centre_susceptible = T1, gradient = 0.1, intensity = gamma , gamma=gamma)


hybrid_model_time_vec = np.zeros(total_repeats, dtype=np.float64)
hybrid_model_logistic_time_vec = np.zeros(total_repeats, dtype=np.float64)


for i in range(total_repeats):
    """
    Running over total repeats for the hybrid_model(fixed threshold)
    """
    print(f"Fixed Threshold Simulation {i + 1}/{total_repeats}")
    _, _, _, _, _, _, _ = hybrid_model.run_multiple(total_simulations=total_sims)
    fixed_threshold_time = hybrid_model.total_time
    hybrid_model_time_vec[i] = fixed_threshold_time

for i in range(total_repeats):
    """
    Running over total repeats for the hybrid_model_logistic
    """
    print(f"Logistic Threshold Simulation {i + 1}/{total_repeats}")
    _, _, _, _, _, _, _ = hybrid_model_logistic.run_multiple(total_simulations=total_sims)
    logistic_threshold_time = hybrid_model_logistic.total_time
    hybrid_model_logistic_time_vec[i] = logistic_threshold_time



# Plotting the histograms
plt.figure()
plt.hist(hybrid_model_time_vec, bins=20, alpha=0.5, label="Fixed Threshold")
plt.hist(hybrid_model_logistic_time_vec, bins=50, alpha=0.5, label="Logistic Threshold")
plt.xlabel('Time taken (s)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.legend()
plt.title('Comparison of Efficiency with Different Thresholds', fontsize=16)
plt.show()

