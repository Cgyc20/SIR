import numpy as np
import matplotlib.pyplot as plt
from Modelling import HybridModel, SISimulation, SISimulation_Mean

# Parameters for the simulation
DS_0 = 300     # Initial number of discrete Susceptible individuals
DI_0 = 1       # Initial number of discrete Infected individuals
CS_0 = 0        # Initial number of continuous Susceptible individuals
CI_0 = 0        # Initial number of continuous Infected individuals
k1 = 0.002      # Rate constant for infection
k2 = 0.1        # Rate constant for recovery
dt = 0.2        # Time step for ODE (Ordinary Differential Equations)
tf = 100         # Final time for the simulation
T1 = 80       # Threshold for converting continuous to discrete Infected
T2 = T1         # Threshold for converting continuous to discrete Susceptible
gamma = 1    # Rate of conversion between discrete and continuous populations

total_sims = 500  # Number of simulations to run

# Create an instance of the HybridModel with the specified parameters
hybrid_model = HybridModel(
    DS_0=DS_0, DI_0=DI_0, CS_0=CS_0, CI_0=CI_0,
    k1=k1, k2=k2, dt=dt, tf=tf, T1=T1, T2=T2, gamma=gamma
)

# Create an instance of the SISimulation with the specified parameters

#combined_model = SISimulation(S0=DS_0, I0=DI_0, k1=k1, k2=k2, tf=tf, dt=dt) #THe Moment closure one

combined_model = SISimulation_Mean(S0=DS_0, I0=DI_0, k1=k1, k2=k2, tf=tf, dt=dt) #The mean field closure

# Run multiple simulations using HybridModel to get the average results
timegrid, DS_vector, DI_vector, CS_vector, CI_vector, HS_vector, HI_vector = hybrid_model.run_multiple(total_simulations=total_sims)

# Run the combined model using SISimulation to get the ODE and SSA results
S_ODE, I_ODE, S_stochastic, I_stochastic = combined_model.run_combined(total_simulations=total_sims)


SSA_time = combined_model.total_SSA_time
ODE_time = combined_model.total_ODE_time
hybrid_time = hybrid_model.total_time

print(
    f"Time taken for {total_sims} simulations using ODE: {ODE_time:.4f} seconds\n"
    f"Time taken for {total_sims} simulations using SSA: {SSA_time:.4f} seconds\n"
    f"Time taken for {total_sims} simulations using Hybrid Model: {hybrid_time:.4f} seconds"
)



fig, (ax1,ax2) = plt.subplots(1,2,figsize = (15,6))

# Plot Hybrid Model results
ax1.plot(timegrid, HI_vector, '--',label='Hybrid Model: $D_I + C_I$', color='black')

# Plot ODE results
ax1.plot(timegrid, I_ODE, label='ODE Infected', color='red')

# Plot SSA results
ax1.plot(timegrid, I_stochastic, label='SSA Infected', color='green')

ax2.plot(timegrid,DI_vector, label = 'Discrete Infected',color = 'red')
ax2.plot(timegrid,CI_vector, label = 'Continuous Infected', color = 'blue')
ax2.plot(timegrid,HI_vector, '--',label = 'Combined', color = 'black')
ax2.plot(timegrid,hybrid_model.T1_vector,'--', label = 'Threshold T1', color = 'grey')

# Adding labels
ax1.set_xlabel('Time')
ax1.set_ylabel('Number of Infected Individuals')
ax1.legend()  # Show legend
ax1.grid(True)  # Add grid for better readability

ax2.set_xlabel('Time')
ax2.set_ylabel('Number of Infected Individuals')
ax2.legend()  # Show legend
ax2.grid(True)  # Add grid for better


ax1.set_ylim([0,max(HI_vector.max(),I_ODE.max(),I_stochastic.max(),HI_vector.max())+10])
ax2.set_ylim([0,max(HI_vector.max(),I_ODE.max(),I_stochastic.max(),HI_vector.max())+10])


# Show the
plt.show()

fig, ax = plt.subplots(figsize=(15,6))

# Plot Hybrid Model results
ax.plot(timegrid, HI_vector, '--',label='Hybrid Model: $D_I + C_I$', color='black')

# Plot ODE results
ax.plot(timegrid, I_ODE, label='ODE Infected', color='red')

# Plot SSA results
ax.plot(timegrid, I_stochastic, label='SSA Infected', color='green')

# Plot Discrete and Continuous Infected results
ax.plot(timegrid, DI_vector, label='Discrete Infected', color='red')
ax.plot(timegrid, CI_vector, label='Continuous Infected', color='blue')
ax.plot(timegrid, HI_vector, '--', label='Combined', color='black')
ax.plot(timegrid, hybrid_model.T1_vector, '--', label='Threshold T1', color='grey')

# Adding labels
ax.set_xlabel('Time')
ax.set_ylabel('Number of Infected Individuals')
ax.legend()  # Show legend
ax.grid(True)  # Add grid for better readability

ax.set_ylim([0, max(HI_vector.max(), I_ODE.max(), I_stochastic.max(), HI_vector.max()) + 10])

# Show the plot
plt.show()
