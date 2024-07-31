import numpy as np
import matplotlib.pyplot as plt
from Modelling import HybridModel, SISimulation

# Parameters for the simulation
DS_0 = 120      # Initial number of discrete Susceptible individuals
DI_0 = 2        # Initial number of discrete Infected individuals
CS_0 = 0        # Initial number of continuous Susceptible individuals
CI_0 = 0        # Initial number of continuous Infected individuals
k1 = 0.002      # Rate constant for infection
k2 = 0.1        # Rate constant for recovery
dt = 0.2        # Time step for ODE (Ordinary Differential Equations)
tf = 40         # Final time for the simulation
T1 = 20         # Threshold for converting continuous to discrete Infected
T2 = T1         # Threshold for converting continuous to discrete Susceptible
gamma = 0.5     # Rate of conversion between discrete and continuous populations

total_sims = 200  # Number of simulations to run

# Create an instance of the HybridModel with the specified parameters
hybrid_model = HybridModel(
    DS_0=DS_0, DI_0=DI_0, CS_0=CS_0, CI_0=CI_0,
    k1=k1, k2=k2, dt=dt, tf=tf, T1=T1, T2=T2, gamma=gamma
)

# Create an instance of the SISimulation with the specified parameters
combined_model = SISimulation(S0=DS_0, I0=DI_0, k1=k1, k2=k2, tf=tf, dt=dt)

# Run multiple simulations using HybridModel to get the average results
timegrid, DS_vector, DI_vector, CS_vector, CI_vector, HS_vector, HI_vector = hybrid_model.run_multiple(total_simulations=total_sims)

# Run the combined model using SISimulation to get the ODE and SSA results
S_ODE, I_ODE, S_stochastic, I_stochastic = combined_model.run_combined(total_simulations=total_sims)

# Plot the results
plt.figure(figsize=(12, 6))

# Plot Hybrid Model results
plt.plot(timegrid, HI_vector, label='Hybrid Model: $D_I + C_I$', color='blue')

# Plot ODE results
plt.plot(timegrid, I_ODE, label='ODE Infected', color='red')

# Plot SSA results
plt.plot(timegrid, I_stochastic, label='SSA Infected', color='green')

# Adding labels and title
plt.xlabel('Time')
plt.ylabel('Number of Infected Individuals')
plt.title('Comparison of Infected Individuals Across Models')
plt.legend()  # Show legend
plt.grid(True)  # Add grid for better readability

# Show the plot
plt.show()
