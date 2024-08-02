import numpy as np
import matplotlib.pyplot as plt
from Modelling import HybridModelLogistic, SISimulation, SISimulation_Mean

# Parameters for the simulation
DS_0 = 200     # Initial number of discrete Susceptible individuals
DI_0 = 2        # Initial number of discrete Infected individuals
CS_0 = 0        # Initial number of continuous Susceptible individuals
CI_0 = 0        # Initial number of continuous Infected individuals
k1 = 0.002      # Rate constant for infection
k2 = 0.1        # Rate constant for recovery
dt = 0.2        # Time step for ODE (Ordinary Differential Equations)
tf = 100        # Final time for the simulation
gamma = 0.5     # Rate of conversion between discrete and continuous populations

Threshold_centre_infected = 50
Threshold_centre_suceptible = 100

intensity = 2
gradient = 1

total_sims = 1000  # Number of simulations to run

# Create an instance of the HybridModel with the specified parameters
hybrid_model = HybridModelLogistic(
    DS_0=DS_0, DI_0=DI_0, CS_0=CS_0, CI_0=CI_0,
    k1=k1, k2=k2, dt=dt, tf=tf, threshold_centre_infected = Threshold_centre_infected, threshold_centre_susceptible = Threshold_centre_suceptible, gradient = gradient, intensity = intensity , gamma=gamma)

# Create an instance of the SISimulation with the specified parameters
combined_model = SISimulation_Mean(S0=DS_0, I0=DI_0, k1=k1, k2=k2, tf=tf, dt=dt)  # The mean field closure

# Run multiple simulations using HybridModel to get the average results
timegrid, DS_vector, DI_vector, CS_vector, CI_vector, HS_vector, HI_vector = hybrid_model.run_multiple(total_simulations=total_sims)

# Run the combined model using SISimulation to get the ODE and SSA results
S_ODE, I_ODE, S_stochastic, I_stochastic = combined_model.run_combined(total_simulations=total_sims)

SSA_time = combined_model.total_SSA_time
ODE_time = combined_model.total_ODE_time
hybrid_time = hybrid_model.total_time

print(f"Time taken for {total_sims} simulations using ODE: {ODE_time:.2f} seconds")
print(f"Time taken for {total_sims} simulations using SSA: {SSA_time:.2f} seconds")
print(f"Time taken for {total_sims} simulations using Hybrid Model: {hybrid_time:.2f} seconds")

# Evaluate logistic function for a range of discrete infected individuals



alpha_vector = np.zeros((len(timegrid), 4), dtype=np.float64)

# Focus on forward reaction! i.e., from Discrete to Continuous
for i in range(len(timegrid)):
    alpha_fI, alpha_bI, alpha_fS, alpha_bS = hybrid_model.logistic_function(DS_vector[i], DI_vector[i], CS_vector[i], CI_vector[i])
    alpha_vector[i] = [alpha_fI, alpha_bI, alpha_fS, alpha_bS]

# Plot results
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Top left plot: Infected comparison
axs[0, 0].plot(timegrid, HI_vector, '--', label='Hybrid Model: $D_I + C_I$', color='black')
axs[0, 0].plot(timegrid, I_ODE, label='ODE Infected', color='red')
axs[0, 0].plot(timegrid, I_stochastic, label='SSA Infected', color='green')

axs[0, 0].set_xlabel('Time')
axs[0, 0].set_ylabel('Number of Infected Individuals')
axs[0, 0].legend()
axs[0, 0].grid(True)
axs[0, 0].set_title('Infected: SSA vs ODE vs Hybrid')

# Top right plot: Susceptible comparison
axs[0, 1].plot(timegrid, HS_vector, '--', label='Hybrid Model: $D_S + C_S$', color='black')
axs[0, 1].plot(timegrid, S_ODE, label='ODE Susceptible', color='red')
axs[0, 1].plot(timegrid, S_stochastic, label='SSA Susceptible', color='green')
axs[0, 1].set_xlabel('Time')
axs[0, 1].set_ylabel('Number of Susceptible Individuals')
axs[0, 1].legend()
axs[0, 1].grid(True)
axs[0, 1].set_title('Susceptible: SSA vs ODE vs Hybrid')

# Bottom left plot: Hybrid Infected
axs[1, 0].plot(timegrid, DI_vector, label='Discrete Infected', color='red')
axs[1, 0].plot(timegrid, CI_vector, label='Continuous Infected', color='blue')
axs[1, 0].plot(timegrid, HI_vector, '--', label='Combined', color='black')
axs[1, 0].plot(timegrid, hybrid_model.threshold_I_vector,'--', label='Threshold centre', color='grey')
axs[1, 0].set_xlabel('Time')
axs[1, 0].set_ylabel('Number of Infected Individuals')
axs[1, 0].legend()
axs[1, 0].grid(True)
axs[1, 0].set_title('Hybrid Model: Infected')

# Bottom right plot: Hybrid Susceptible
axs[1, 1].plot(timegrid, DS_vector, label='Discrete Susceptible', color='red')
axs[1, 1].plot(timegrid, CS_vector, label='Continuous Susceptible', color='blue')
axs[1, 1].plot(timegrid, HS_vector, '--', label='Combined', color='black')
axs[1, 1].plot(timegrid, hybrid_model.threshold_S_vector,'--', label='Threshold centre', color='grey')
axs[1, 1].set_xlabel('Time')
axs[1, 1].set_ylabel('Number of Susceptible Individuals')
axs[1, 1].legend()
axs[1, 1].grid(True)
axs[1, 1].set_title('Hybrid Model: Susceptible')

plt.tight_layout()
plt.show()




# Plot logistic function results
plt.figure(figsize=(10, 6))
plt.plot(timegrid, alpha_vector[:, 0],'--', label='alpha_fI')
plt.plot(timegrid, alpha_vector[:, 1], label='alpha_bI')
plt.plot(timegrid, alpha_vector[:, 2], label='alpha_fS')
plt.plot(timegrid, alpha_vector[:, 3],'--', label='alpha_Bs')  # Repeated due to typo in original code
plt.xlabel('Time (days)')
plt.ylabel('Logistic Function Value')
plt.legend()
plt.grid(True)
plt.show()
