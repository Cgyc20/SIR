import numpy as np
import matplotlib.pyplot as plt
import random
import tqdm

# Parameters for both simulations
S0 = 400 # Initial discrete Susceptible
I0 = 1 # Initial discrete Infected

S02 = S0**2
I02 = I0**2
SI0 = S0*I0


k1 = 0.002 # First rate constant
k2 = 0.1 # Second rate
tf = 50 # Final time
dt = 0.1 # Time step
num_points = int(tf / dt) + 1  # Number of time points in the simulation
timegrid = np.linspace(0, tf, num_points, dtype=np.float64)  # Time points
number_molecules = 2 

# Stoichiometric matrix for reactions
S_matrix = np.array([[-1, 1],
                     [0, -1]], dtype=int)

# Function to compute propensities
def compute_propensities(states):
    S, I = states
    alpha_1 = k1 * S * I
    alpha_2 = k2 * I
    return np.array([alpha_1, alpha_2], dtype=float)

# Function to perform reactions
def perform_reaction(index, states):
    states += S_matrix[index]
    states[0] = np.max(states[0], 0)
    states[1] = np.max(states[1], 0)
    return states

# Function to perform one step of Gillespie algorithm
def gillespie_step(alpha_cum, alpha0, states):
    r2 = random.uniform(0, 1)
    index = next(i for i, alpha in enumerate(alpha_cum / alpha0) if r2 <= alpha)
    return perform_reaction(index, states)

# Function to run Gillespie simulation
def run_gillespie_simulation():
    data_table_cum = np.zeros((num_points, number_molecules), dtype=np.float64)
    total_simulations = 200

    for _ in tqdm.tqdm(range(total_simulations)):
        t = 0
        old_time = t
        data_table = np.zeros((num_points, number_molecules), dtype=np.float64)
        states = np.array([S0, I0], dtype=int)

        while t < tf:
            alpha_list = compute_propensities(states)
            alpha0 = sum(alpha_list)
            if alpha0 == 0:
                break
            alpha_cum = np.cumsum(alpha_list)
            tau = np.log(1 / random.uniform(0, 1)) / alpha0
            states = gillespie_step(alpha_cum, alpha0, states)
            old_time = t
            t += tau

            ind_before = np.searchsorted(timegrid, old_time, 'right')
            ind_after = np.searchsorted(timegrid, t, 'left')
            for index in range(ind_before, min(ind_after + 1, num_points)):
                data_table[index, :] = states

        data_table_cum += data_table

    data_table_cum /= total_simulations
    return data_table_cum

# Differential equations for deterministic model
def differential(S, I, I2, SI, S2):
    DSIDT = k1*((S2*SI**2)/(I*S**2)-((SI**2)*I2)/(S*I**2)-SI)-k2*SI
    DS2DT = k1*(SI-2*((SI**2)*S2)/((S**2)*I))
    DI2DT = k1*(2*((SI**2)*I2)/(S*I**2)+SI)+k2*(I-2*I2)

    dSdt = -k1 * SI
    dIdt = k1 * SI - k2 * I

    return dSdt, dIdt, DSIDT, DS2DT, DI2DT

# Runge-Kutta step function
def RK4_step(S, I, I2, SI, S2, dt):
    k1_S, k1_I, k1_SI, k1_S2, k1_I2 = differential(S, I, I2, SI, S2)
    k2_S, k2_I, k2_SI, k2_S2, k2_I2 = differential(S + k1_S * dt / 2, I + k1_I * dt / 2, I2 + k1_I2 * dt / 2, SI + k1_SI * dt / 2, S2 + k1_S2 * dt / 2)
    k3_S, k3_I, k3_SI, k3_S2, k3_I2 = differential(S + k2_S * dt / 2, I + k2_I * dt / 2, I2 + k2_I2 * dt / 2, SI + k2_SI * dt / 2, S2 + k2_S2 * dt / 2)
    k4_S, k4_I, k4_SI, k4_S2, k4_I2 = differential(S + k3_S * dt, I + k3_I * dt, I2 + k3_I2 * dt, SI + k3_SI * dt, S2 + k3_S2 * dt)
    
    S_next = S + dt * (k1_S + 2 * k2_S + 2 * k3_S + k4_S) / 6
    I_next = I + dt * (k1_I + 2 * k2_I + 2 * k3_I + k4_I) / 6
    SI_next = SI + dt*(k1_SI + 2 * k2_SI + 2 * k3_SI + k4_SI )/6
    S2_next = S2 + dt*(k1_S2 + 2 * k2_S2 + 2 * k3_S2 + k4_S2 )/6
    S3_next = I2 + dt*(k1_I2 + 2 * k2_I2 + 2 * k3_I2 + k4_I2 )/6

    
    return S_next, I_next, SI_next, S2_next, S3_next

# Function to run deterministic simulation
def run_deterministic_simulation():
    S = np.zeros(num_points)
    I = np.zeros(num_points)
    
    S2 = np.zeros(num_points)
    I2 = np.zeros(num_points)
    SI = np.zeros(num_points)

    S[0] = S0
    I[0] = I0
    SI[0] = SI0
    S2[0] = S02
    I2[0] = I02
    
    for i in range(1, num_points):
        S[i], I[i], SI[i], S2[i], I2[i] = RK4_step(S[i - 1], I[i - 1], I2[i-1], SI[i-1], S2[i-1], dt)
        
    return S, I



# Run simulations
data_table_gillespie = run_gillespie_simulation()
S_det, I_det = run_deterministic_simulation()

# Plot results
plt.figure()

# Plot for Susceptible (S)
plt.subplot(2, 1, 1)
plt.plot(timegrid, data_table_gillespie[:, 0], label='S (Gillespie)')
plt.plot(timegrid, S_det, label='S (Deterministic)', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('Susceptible')
plt.legend()

# Plot for Infected (I)
plt.subplot(2, 1, 2)
plt.plot(timegrid, data_table_gillespie[:, 1], label='I (Gillespie)')
plt.plot(timegrid, I_det, label='I (Deterministic)', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('Infected')
plt.legend()

plt.tight_layout()
plt.show()
