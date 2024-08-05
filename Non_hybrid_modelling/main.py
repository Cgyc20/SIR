import numpy as np
import matplotlib.pyplot as plt
from ODE_class import ODE_class  # Make sure the ODE_class is saved in a file named 'ode_class.py'
from SDE_class import StochasticSIR  # Make sure the StochasticSIR class is saved in a file named 'stochastic_sir.py'

# Parameters for ODE model
S0 = 200   # Initial susceptible proportion
I0 = 1     # Initial infected proportion
k1 = 0.002 # First rate constant
k2 = 0.1   # Second rate constant
dt = 0.1   # Time step
tf = 40    # Final time

# Create an instance of ODE_class
ode = ODE_class(S0, I0, k1, k2, dt, tf)

# Run the mean field model
S_mean, I_mean = ode.run_mean_field()

# Run the Kirkwood model
S_kirk, I_kirk, SI_kirk, S2_kirk, I2_kirk = ode.run_kirkwood()

# Parameters for Stochastic model
number_molecules = 2
total_simulations = 100

# Create an instance of StochasticSIR
stochastic = StochasticSIR(S0, I0, k1, k2, tf, dt, number_molecules, total_simulations)

# Run the stochastic simulations
data_table_cum = stochastic.run_all_simulations()

# Extract results for plotting
S_stoch = data_table_cum[:, 0]
I_stoch = data_table_cum[:, 1]

# Plot the results
plt.figure(figsize=(8, 6))

# Plot mean field results
plt.subplot(3, 1, 1)
plt.plot(ode.timegrid, S_mean, label='S (Mean Field)', linestyle='--')
plt.plot(ode.timegrid, I_mean, label='I (Mean Field)', linestyle='-')
plt.title('Mean Field Model')
plt.xlabel('Time')
plt.ylabel('Proportion')
plt.legend()

# Plot Kirkwood results
plt.subplot(3, 1, 2)
plt.plot(ode.timegrid, S_kirk, label='S (Kirkwood)', linestyle='--')
plt.plot(ode.timegrid, I_kirk, label='I (Kirkwood)', linestyle='-')
plt.title('Kirkwood Model')
plt.xlabel('Time')
plt.ylabel('Proportion')
plt.legend()

# Plot Stochastic simulation results
plt.subplot(3, 1, 3)
plt.plot(stochastic.timegrid, S_stoch, label='S (Stochastic)', linestyle='--')
plt.plot(stochastic.timegrid, I_stoch, label='I (Stochastic)', linestyle='-')
plt.title('Stochastic Simulation')
plt.xlabel('Time')
plt.ylabel('Proportion')
plt.legend()

plt.tight_layout()
plt.show()
