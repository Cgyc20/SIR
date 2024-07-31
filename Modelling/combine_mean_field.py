import numpy as np
import random
import tqdm
import time 

class SISimulation_Mean:
    def __init__(self, S0=400, I0=2, k1=0.002, k2=0.1, tf=50, dt=0.1):
        """
        Initialize the simulation parameters and state variables.
        
        Parameters:
        S0 (int): Initial susceptible population.
        I0 (int): Initial infected population.
        k1 (float): Infection rate.
        k2 (float): Recovery rate.
        tf (float): Final time of the simulation.
        dt (float): Time step for the simulation.
        """
        self.S0 = S0
        self.I0 = I0
        self.k1 = k1
        self.k2 = k2
        self.tf = tf
        self.dt = dt
        

        
        self.num_points = int(tf / dt) + 1  # Total time steps
        self.timegrid = np.linspace(0, tf, self.num_points, dtype=np.float64)  # Time grid
        self.number_molecules = 2
        self.states = np.array([S0, I0], dtype=np.float64)  # Initial states
        self.S_matrix = np.array([[-1, 1], [0, -1]], dtype=int)  # Stoichiometric matrix
    
    def compute_propensities(self, states):
        """
        Compute the propensity functions for the reactions.
        
        Parameters:
        states (numpy array): The current states of the system.
        
        Returns:
        numpy array: Propensity functions [alpha1, alpha2].
        """
        S, I = states
        alpha_1 = self.k1 * S * I
        alpha_2 = self.k2 * I
        return np.array([alpha_1, alpha_2], dtype=float)
    
    def perform_reaction(self, index, states):
        """
        Update the states based on the selected reaction.
        
        Parameters:
        index (int): Index of the reaction.
        states (numpy array): The current states of the system.
        
        Returns:
        numpy array: Updated states after the reaction.
        """
        states += self.S_matrix[index]
        states[0] = np.max(states[0], 0)
        states[1] = np.max(states[1], 0)
        return states
    
    def gillespie_step(self, alpha_cum, alpha0, states):
        """
        Perform one step of the Gillespie algorithm.
        
        Parameters:
        alpha_cum (numpy array): Cumulative sum of the propensity functions.
        alpha0 (float): Sum of the propensity functions.
        states (numpy array): The current states of the system.
        
        Returns:
        numpy array: Updated states after the reaction.
        """
        r2 = random.uniform(0, 1)
        index = next(i for i, alpha in enumerate(alpha_cum / alpha0) if r2 <= alpha)
        return self.perform_reaction(index, states)
    
    def run_gillespie_simulation(self, total_simulations=200):
        """
        Run the Gillespie stochastic simulation algorithm.
        
        Parameters:
        total_simulations (int): Number of simulations to run.
        
        Returns:
        numpy array: Averaged states over all simulations.
        """
        data_table_cum = np.zeros((self.num_points, self.number_molecules), dtype=np.float64)
        print("Running multiple simulations of the SSA (Non-hybrid)...")
        for _ in tqdm.tqdm(range(total_simulations)):
            t = 0
            data_table = np.zeros((self.num_points, self.number_molecules), dtype=np.float64)
            states = np.array([self.S0, self.I0], dtype=int)
            data_table[0, :] = states
            while t < self.tf:
                alpha_list = self.compute_propensities(states)
                alpha0 = sum(alpha_list)
                if alpha0 == 0:
                    break
                alpha_cum = np.cumsum(alpha_list)
                tau = np.log(1 / random.uniform(0, 1)) / alpha0
                states = self.gillespie_step(alpha_cum, alpha0, states)
                old_time = t
                t += tau

                ind_before = np.searchsorted(self.timegrid, old_time, 'right')
                ind_after = np.searchsorted(self.timegrid, t, 'left')
                for index in range(ind_before, min(ind_after + 1, self.num_points)):
                    data_table[index, :] = states

            data_table_cum += data_table

        data_table_cum /= total_simulations
        return data_table_cum

    def differential(self, S, I):
        """
        Calculate the derivatives for the ODE system using the Kirkwood approximation moment closure.
        
        Parameters:
        S (float): Susceptible population.
        I (float): Infected population.
        Returns:
        tuple: Derivatives (dS/dt, dI/dt.
        """
    
        dSdt = -self.k1 * S*I
        dIdt = self.k1 * S*I - self.k2 * I

        return dSdt, dIdt
    
    def RK4_step(self, S, I):
        """
        Perform one step of the Runge-Kutta 4th order (RK4) method.
        
        Parameters:
        S (float): Susceptible population.
        I (float): Infected population.

        Returns:
        tuple: Updated values (S_next, I_next).
        """
       
        k1_S, k1_I = self.differential(S, I)
        k2_S, k2_I = self.differential(S + 0.5 * k1_S * self.dt, I + 0.5 * k1_I * self.dt)
        k3_S, k3_I = self.differential(S + 0.5 * k2_S * self.dt, I + 0.5 * k2_I * self.dt)
        k4_S, k4_I = self.differential(S + k3_S * self.dt, I + k3_I * self.dt)

        S_next = S + (k1_S + 2 * k2_S + 2 * k3_S + k4_S) * self.dt / 6
        I_next = I + (k1_I + 2 * k2_I + 2 * k3_I + k4_I) * self.dt / 6


        return S_next, I_next
    
    def run_deterministic_simulation(self):
        """
        Run the deterministic solver over the total number of time points.
        
        Returns:
        tuple: Time series of S and I.
        """
        S = np.zeros(self.num_points)  # Susceptible vector
        I = np.zeros(self.num_points)  # Infected vector
        
        S[0] = self.S0
        I[0] = self.I0
        
        # Main loop over the time steps
        for i in range(1, self.num_points):
            S[i], I[i] = self.RK4_step(S[i - 1], I[i - 1])
        
        return S, I
    
    def run_combined(self, total_simulations):
        """
        Run the combined deterministic and stochastic models.
        
        Parameters:
        total_simulations (int): Number of stochastic simulations to run.
        
        Returns:
        tuple: Time series of S, I from deterministic model, and averaged states from stochastic model.
        """
        ODE_start_time = time.time()
        S, I = self.run_deterministic_simulation()
        self.total_ODE_time = time.time() - ODE_start_time

        SSA_start_time = time.time()
        data_table_cum = self.run_gillespie_simulation(total_simulations=total_simulations)
        self.total_SSA_time = time.time() - SSA_start_time

        S_stochastic  = data_table_cum[:, 0]
        I_stochastic = data_table_cum[:,1]

        return S, I, S_stochastic, I_stochastic
