import numpy as np
import matplotlib.pyplot as plt
from Modelling import HybridModel, SISimulation_Mean
from matplotlib.ticker import MaxNLocator


# Parameters for the simulation
DS_0 = 500       # Initial number of discrete Susceptible individuals
DI_0 = 1         # Initial number of discrete Infected individuals
CS_0 = 0         # Initial number of continuous Susceptible individuals
CI_0 = 0         # Initial number of continuous Infected individuals
k1 = 0.002       # Rate constant for infection
k2 = 0.1         # Rate constant for recovery
dt = 0.2         # Time step for ODE (Ordinary Differential Equations)
tf = 60          # Final time for the simulation
gamma = 2        # Rate of conversion between discrete and continuous populations
total_sims = 1000  # Number of simulations to run



starting_threshold, ending_threshold = map(float,input("Enter the starting and ending threshold values separated by a space: ").split())
print(starting_threshold)
print(f"Running simulation from T1 = {starting_threshold} - {ending_threshold}")
# Threshold vector for conversion
threshold_vector = np.arange(starting_threshold,ending_threshold , 1, dtype=int)

# Arrays to store results
mean_hybrid_error_vector = np.zeros(len(threshold_vector))
max_error =np.zeros(len(threshold_vector))

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
    #hybrid_error_vector = np.abs(HI_vector - I_stochastic) / np.abs(I_stochastic)
    hybrid_error_vector = np.abs(HI_vector - I_stochastic) 


    mean_hybrid_error_vector[i] = np.mean(hybrid_error_vector)
    max_error[i] = np.max(hybrid_error_vector)

# Plot the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot mean relative error on the first subplot
ax1.plot(threshold_vector, mean_hybrid_error_vector, marker='o', color='black', markerfacecolor='blue', linestyle=':', markersize=8, label=r'$\frac{1}{N}\sum_{i=0}^{N-1}|I_j-(I_H)_j|$')

# Label the axes and add a title
ax1.set_xlabel('Threshold for conversion', fontsize=12)
ax1.set_ylabel('Error', fontsize=12)

# Ensure x-axis only displays integers
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.legend(title_fontsize='13', fontsize='14')

# Add legend and grid
ax1.legend(title_fontsize='13', fontsize='14')
ax1.grid(True, which='both', linestyle='--', linewidth=0.7)

# Plot maximum error on the second subplot
ax2.plot(threshold_vector, max_error, marker='o', color='black', markerfacecolor='blue', linestyle=':', markersize=8, label=r'$\max_{0 \leq i < N} | I_j - (I_H)_j |$')

# Label the axes and add a title
ax2.set_xlabel('Threshold for conversion', fontsize=12)
ax2.set_ylabel('Error', fontsize=12)

# Ensure x-axis only displays integers
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

# Add legend and grid
ax2.legend(title_fontsize='13', fontsize='11')
ax2.grid(True, which='both', linestyle='--', linewidth=0.7)
ax2.legend(title_fontsize='13', fontsize='14')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

