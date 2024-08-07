import numpy as np
import sys
import os

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Modelling import HybridModel, SISimulation_Mean  # type: ignore

# Parameters for the simulation
DS_0 = 250       # Initial number of discrete Susceptible individuals
DI_0 = 1         # Initial number of discrete Infected individuals
CS_0 = 0         # Initial number of continuous Susceptible individuals
CI_0 = 0         # Initial number of continuous Infected individuals
k1 = 0.002       # Rate constant for infection
k2 = 0.1         # Rate constant for recovery
dt = 0.2         # Time step for ODE (Ordinary Differential Equations)
tf = 60          # Final time for the simulation
gamma = 1      # Rate of conversion between discrete and continuous populations

# Get user inputs with error handling
try:
    starting_threshold, ending_threshold = map(float, input("Enter the starting and ending threshold values separated by a space: ").split())
    total_sims = int(input("Enter the number of simulations to run: "))
except ValueError:
    print("Invalid input. Please enter numeric values.")
    sys.exit(1)

print(f"Running simulation from T1 = {starting_threshold} to {ending_threshold}")
print(f"{total_sims} total simulations")

# Threshold vector for conversion
threshold_vector = np.arange(starting_threshold, ending_threshold, 1, dtype=int)

# Arrays to store results
mean_hybrid_error_vector = np.zeros(len(threshold_vector))

# Create an instance of SISimulation_Mean and run the combined model
combined_model = SISimulation_Mean(S0=DS_0, I0=DI_0, k1=k1, k2=k2, tf=tf, dt=dt)
S_ODE, I_ODE, S_stochastic, I_stochastic = combined_model.run_combined(total_simulations=total_sims)

Error_matrix = np.zeros((len(threshold_vector), len(combined_model.timegrid)), dtype=np.float64)

# Loop over the threshold values
for i, T1 in enumerate(threshold_vector):
    T2 = T1
    print(f"Iteration: {i}, Threshold: {T1}")

    # Create an instance of the HybridModel with the specified parameters
    hybrid_model = HybridModel(
        DS_0=DS_0, DI_0=DI_0, CS_0=CS_0, CI_0=CI_0,
        k1=k1, k2=k2, dt=dt, tf=tf, T1=T1, T2=T2, gamma=gamma
    )

    # Run multiple simulations using HybridModel to get the average results
    _, _, _, _, _, _, HI_vector = hybrid_model.run_multiple(total_simulations=total_sims)
    
    # Calculate the error
    hybrid_error_vector = np.abs(HI_vector - I_stochastic)
    Error_matrix[i, :] = hybrid_error_vector

# Save the data matrix and threshold matrix
np.save("Error_matrix.npy", Error_matrix)
np.save("Threshold_vector.npy", threshold_vector)