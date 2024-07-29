import numpy as np
import matplotlib.pyplot as plt
import random
import tqdm

# Parameters for both simulations
S0 = 400 # Initial discrete Susceptible
I0 = 1  # Initial discrete Infected
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
    total_simulations = 100

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
def differential(S, I):
    dSdt = -k1 * S * I
    dIdt = k1 * S * I - k2 * I
    return dSdt, dIdt

# Runge-Kutta step function
def RK4_step(S, I, dt):
    k1_S, k1_I = differential(S, I)
    k2_S, k2_I = differential(S + k1_S * dt / 2, I + k1_I * dt / 2)
    k3_S, k3_I = differential(S + k2_S * dt / 2, I + k2_I * dt / 2)
    k4_S, k4_I = differential(S + k3_S * dt, I + k3_I * dt)
    S_next = S + dt * (k1_S + 2 * k2_S + 2 * k3_S + k4_S) / 6
    I_next = I + dt * (k1_I + 2 * k2_I + 2 * k3_I + k4_I) / 6
    return S_next, I_next

# Function to run deterministic simulation
def run_deterministic_simulation():
    S = np.zeros(num_points)
    I = np.zeros(num_points)
    S[0] = S0
    I[0] = I0

    for i in range(1, num_points):
        S[i], I[i] = RK4_step(S[i-1], I[i-1], dt)

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
