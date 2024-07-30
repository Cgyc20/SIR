import numpy as np
import matplotlib.pyplot as plt
from hybrid_model_class import HybridModel
from combine_class import SISimulation 


DS_0 = 120  # Initial discrete Susceptible
DI_0 = 2  # Initial discrete Infected
CS_0 = 0   # Initial continuous Susceptible
CI_0 = 0   # Initial continuous Infected
k1 = 0.002 # First rate constant
k2 = 0.1   # Second rate
dt = 0.2   # Time step (For ODE)
tf = 40    # Final time
T1 = 20    # Threshold for conversion (Infected)
T2 = T1    # Threshold for conversion (Susceptible)
gamma = 0.5 # The rate of conversion 
number_molecules = 4 # The total molecules (two discrete, two continuous)
total_sims = 200


# Create instances of HybridModel and SISimulation with predefined rates
Hybrid_Model = HybridModel(DS_0=DS_0, DI_0=DI_0, CS_0=CS_0, CI_0=CI_0, k1=k1, k2=k2, dt=dt, tf=tf, T1=T1, T2=T2, gamma=gamma)
combined_model = SISimulation(S0=DS_0, I0=DI_0, k1=k1, k2=k2, tf=tf, dt=dt)

# Run multiple simulations using HybridModel
timegrid, data_table_cum, combined_vector = Hybrid_Model.run_multiple(total_simulations=total_sims)

# Run the combined model using SISimulation
S, I, data_table_combined = combined_model.run_combined(total_simulations=total_sims)

# Plot the results
plt.figure()
plt.plot(timegrid, combined_vector[:, 1], label='Hybrid $D_I+C_I$')
plt.plot(timegrid, I, label='ODE Infected')
plt.plot(timegrid,data_table_combined[:,1], label = 'SSA $I$')
plt.legend()
plt.show()




# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# ax1.plot(timegrid, data_table_combined[:, 0], label='$D_I$ Discrete')
# ax1.plot(timegrid, data_table_combined[:, 1], label='$C_I$ Continuous')

# ax1.plot(timegrid, combined[:, 1], label='$C_I+D_I$ Combined', color='black', linestyle='--')
# ax1.set_xlabel('days')
# ax1.set_ylabel('Number infected')
# ax1.legend()
# ax1.grid(True)
# ax1.set_title('Current Infected Over Time')

# ax2.plot(timegrid, data_table_cum[:, 0], label='$D_S$ Discrete')
# ax2.plot(timegrid, data_table_cum[:, 1], label='$C_S$ Continuous')
# ax2.plot(timegrid, combined[:, 0], label='$C_S+D_S$ Combined', color='black', linestyle='--')
# ax2.set_xlabel('days')
# ax2.set_ylabel('Number susceptible')
# ax2.legend()
# ax2.grid(True)
# ax2.set_title('Susceptible Over Time')

# plt.tight_layout()
# plt.show()




# threshold = np.ones_like(timegrid)*T1

# """Plotting the Hybrid Hybrid_model"""

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# ax1.plot(timegrid, data_table_cum[:, 1], label='$D_I$ Discrete')
# ax1.plot(timegrid, data_table_cum[:, 3], label='$C_I$ Continuous')
# ax1.plot(timegrid, combined[:, 1], label='$C_I+D_I$ Combined', color='black', linestyle='--')
# ax1.plot(timegrid, threshold, '--', label='Conversion Threshold')
# ax1.set_xlabel('days')
# ax1.set_ylabel('Number infected')
# ax1.legend()
# ax1.grid(True)
# ax1.set_title('Current Infected Over Time')

# ax2.plot(timegrid, data_table_cum[:, 0], label='$D_S$ Discrete')
# ax2.plot(timegrid, data_table_cum[:, 2], label='$C_S$ Continuous')
# ax2.plot(timegrid, combined[:, 0], label='$C_S+D_S$ Combined', color='black', linestyle='--')
# ax2.set_xlabel('days')
# ax2.set_ylabel('Number susceptible')
# ax2.legend()
# ax2.grid(True)
# ax2.set_title('Susceptible Over Time')

# plt.tight_layout()
# plt.show()


