import numpy as np
import matplotlib.pyplot as plt
from Modelling import HybridModel, SISimulation, SISimulation_Mean
import tqdm


# Parameters for the simulation
DS_0 = 150       # Initial number of discrete Infected individuals
CS_0 = 0        # Initial number of continuous Susceptible individuals
CI_0 = 0        # Initial number of continuous Infected individuals
k1 = 0.002      # Rate cons tant for infection
k2 = 0.1        # Rate constant for recovery
dt = 0.2        # Time step for ODE (Ordinary Differential Equations)
tf = 60         # Final time for the simulation
T1 = 25       # Threshold for converting continuous to discrete Infected
T2 = T1         # Threshold for converting continuous to discrete Susceptible
gamma = 2    # Rate of conversion between discrete and continuous populations
total_sims = 2000  # Number of simulations to run

discrete_infected_vector = np.arange(1,5,1)

accuracy = np.zeros((len(discrete_infected_vector),3),dtype = np.float64)

mean_ode_error_vector = np.zeros(len(discrete_infected_vector))
mean_hybrid_error_vector = np.zeros(len(discrete_infected_vector))

for i in range(len(discrete_infected_vector)):
    print(i)
    DI_0 = discrete_infected_vector[i]


    combined_model = SISimulation_Mean(S0=DS_0, I0=DI_0, k1=k1, k2=k2, tf=tf, dt=dt)
    # Run the combined model using SISimulation to get the ODE and SSA results
    S_ODE, I_ODE, S_stochastic, I_stochastic = combined_model.run_combined(total_simulations=total_sims)

    # Create an instance of the HybridModel with the specified parameters
    hybrid_model = HybridModel(
        DS_0=DS_0, DI_0=DI_0, CS_0=CS_0, CI_0=CI_0,
        k1=k1, k2=k2, dt=dt, tf=tf, T1=T1, T2=T2, gamma=gamma
    )

    # Create an instance of the SISimulation with the specified parameters

    # Run multiple simulations using HybridModel to get the average results
    timegrid, DS_vector, DI_vector, CS_vector, CI_vector, HS_vector, HI_vector = hybrid_model.run_multiple(total_simulations=total_sims)

    hybrid_error_vector = np.abs(HI_vector - I_stochastic)/np.abs(I_stochastic)
    ODE_error = np.abs(I_ODE - I_stochastic)/np.abs(I_stochastic)

    mean_hybrid_error_vector[i] = np.mean(hybrid_error_vector)
    mean_ode_error_vector[i] = np.mean(ODE_error)


plt.figure(figsize=(10, 6))

# Define colors and line styles
colors = ['#1f77b4', '#ff7f0e']  # Colors suitable for colorblind viewers
linestyles = ['-', '--']  # Different line styles

# Plot each series with the same marker but different colors and line styles
plt.plot(discrete_infected_vector, mean_hybrid_error_vector, marker='o', color=colors[0], linestyle=linestyles[0], markersize=8, label='Hybrid error')
plt.plot(discrete_infected_vector, mean_ode_error_vector, marker='o', color=colors[1], linestyle=linestyles[1], markersize=8, label='ODE error')

# Label the axes
plt.xlabel('Initial number of discrete Infected individuals', fontsize=12)
plt.ylabel('Mean relative error', fontsize=12)

# Add title
plt.title('Accuracy of Hybrid Model', fontsize=14)

# Add legend
plt.legend(title='Error Type', title_fontsize='13', fontsize='11')

# Add grid
plt.grid(True, which='both', linestyle='--', linewidth=0.7)

# Display the plot
plt.tight_layout()
plt.show()

