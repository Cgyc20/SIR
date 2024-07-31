import numpy as np
import matplotlib.pyplot as plt
from Modelling import HybridModel, SISimulation_Mean
from matplotlib.ticker import MaxNLocator

# Parameters for the simulation
DS_0 = 200       # Initial number of discrete Susceptible individuals
DI_0 = 1         # Initial number of discrete Infected individuals
CS_0 = 0         # Initial number of continuous Susceptible individuals
CI_0 = 0         # Initial number of continuous Infected individuals
k1 = 0.002       # Rate constant for infection
k2 = 0.1         # Rate constant for recovery
dt = 0.2         # Time step for ODE (Ordinary Differential Equations)
tf = 60          # Final time for the simulation
gamma = 2        # Rate of conversion between discrete and continuous populations
total_sims = 5000  # Number of simulations to run

# Threshold vector for conversion
threshold_vector = np.arange(0, 25, 1, dtype=int)

# Arrays to store results
mean_hybrid_error_vector = np.zeros(len(threshold_vector))

# Create an instance of SISimulation_Mean and run the combined model
combined_model = SISimulation_Mean(S0=DS_0, I0=DI_0, k1=k1, k2=k2, tf=tf, dt=dt)
S_ODE, I_ODE, S_stochastic, I_stochastic = combined_model.run_combined(total_simulations=total_sims)

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
    timegrid, DS_vector, DI_vector, CS_vector, CI_vector, HS_vector, HI_vector = hybrid_model.run_multiple(total_simulations=total_sims)

    # Calculate the mean hybrid error
    hybrid_error_vector = np.abs(HI_vector - I_stochastic) / np.abs(I_stochastic)
    mean_hybrid_error_vector[i] = np.mean(hybrid_error_vector)

# Plot the results
plt.plot(threshold_vector, mean_hybrid_error_vector, marker='o', color='#1f77b4', linestyle='-', markersize=8, label='Hybrid error')

# Label the axes and add a title
plt.xlabel('Threshold for conversion', fontsize=12)
plt.ylabel('Mean relative error', fontsize=12)
plt.title('Accuracy of Hybrid Model', fontsize=14)

# Ensure x-axis only displays integers
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

# Add legend and grid
plt.legend(title_fontsize='13', fontsize='11')
plt.grid(True, which='both', linestyle='--', linewidth=0.7)

# Display the plot
plt.tight_layout()
plt.show()
