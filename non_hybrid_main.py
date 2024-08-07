import matplotlib.pyplot as plt
from Non_hybrid_modelling import StochasticSIR, ODE_mean_class, ODE_kirkwood  # Make sure the ODE_class is saved in a file named 'ode_class.py'


# Parameters for ODE model
S0 = 120 # Initial susceptible proportion
I0 = 4     # Initial infected proportion
k1 = 0.002 # First rate constant
k2 = 0.1   # Second rate constant
dt = 0.1   # Time step
tf = 100    # Final time

# Create an instance of ODE_class
ode_mean_field = ODE_mean_class(S0, I0, k1, k2, dt, tf)

# Create an instance of ODE_class
ode_kirkwood = ODE_kirkwood(S0, I0, k1, k2, dt, tf)

# Run the mean field model
S_mean, I_mean = ode_mean_field.run_mean_field()

# Run the Kirkwood model
S_kirk, I_kirk, SI_kirk, S2_kirk, I2_kirk = ode_kirkwood.run_kirkwood()

# Parameters for Stochastic model
number_molecules = 2
total_simulations = 1000

# Create an instance of StochasticSIR
stochastic = StochasticSIR(S0, I0, k1, k2, tf, dt, number_molecules, total_simulations)

# Run the stochastic simulations
data_table_cum = stochastic.run_all_simulations()

# Extract results for plotting

I_stochastic= data_table_cum[:, 1]

# Plot the results

plt.figure(figsize=(10, 8))

# Plotting the data
plt.plot(ode_mean_field.timegrid, I_mean, label='Mean-field ODE', linewidth=2)
plt.plot(stochastic.timegrid, I_stochastic,'--', label='SSA', linewidth=2, color = 'black')
plt.plot(ode_kirkwood.timegrid, I_kirk, color='r', label='Kirkwood closure ODE', linewidth=2)

# Adding titles and labels
plt.title('SIR Model Simulation', fontsize=16)
plt.xlabel('Time (days)', fontsize=14)
plt.ylabel('Proportion of Infected Individuals', fontsize=14)

# Adding grid
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Improving the legend
plt.legend(fontsize=12, loc='best')

# Tight layout for better spacing
plt.tight_layout()

# Display the plot
plt.show()
