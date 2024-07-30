import numpy as np
import matplotlib.pyplot as plt
from hybrid_model_class import HybridModel

DS_0 = 400 #Initial discrete Suceptible
DI_0 = 5  #Initial discrete Infected
CS_0 = 0 #Initial continious Suceptible
CI_0 = 0 #Initial continious Infected
k1 = 0.002 #First rate constant
k2 = 0.1 #Second rate
dt = 0.2 #Time step (For ODE)
tf = 40 #Final time
T1 = 40 #Threshold for conversion (Infected)
T2 =T1 #Threshold for conversion (suceptible)
gamma = 0.5 #The rate of conversion 
number_molecules = 4 #The total molecules (two discrete,two cont)





Model = HybridModel(DS_0 = 400, DI_0 = 5, CS_0 = 0,CI_0 = 0, k1=0.002, k2=0.1, dt=0.2, tf=40, T1=40, T2=40, gamma=0.5)

timegrid,data_table_cum, combined = Model.run_multiple(total_simulations=100)

threshold = np.ones_like(timegrid)*T1

"""Plotting the Hybrid model"""

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

ax1.plot(timegrid, data_table_cum[:, 1], label='$D_I$ Discrete')
ax1.plot(timegrid, data_table_cum[:, 3], label='$C_I$ Continuous')
ax1.plot(timegrid, combined[:, 1], label='$C_I+D_I$ Combined', color='black', linestyle='--')
ax1.plot(timegrid, threshold, '--', label='Conversion Threshold')
ax1.set_xlabel('days')
ax1.set_ylabel('Number infected')
ax1.legend()
ax1.grid(True)
ax1.set_title('Current Infected Over Time')

ax2.plot(timegrid, data_table_cum[:, 0], label='$D_S$ Discrete')
ax2.plot(timegrid, data_table_cum[:, 2], label='$C_S$ Continuous')
ax2.plot(timegrid, combined[:, 0], label='$C_S+D_S$ Combined', color='black', linestyle='--')
ax2.set_xlabel('days')
ax2.set_ylabel('Number susceptible')
ax2.legend()
ax2.grid(True)
ax2.set_title('Susceptible Over Time')

plt.tight_layout()
plt.show()


