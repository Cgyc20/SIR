import numpy as np
import random
import tqdm

class SISimulation:
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
        
        self.S02 = S0**2
        self.I02 = I0**2
        self.SI0 = S0 * I0
        
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

    def differential(self, S, I, I2, SI, S2):
        """
        Calculate the derivatives for the ODE system using the Kirkwood approximation moment closure.
        
        Parameters:
        S (float): Susceptible population.
        I (float): Infected population.
        I2 (float): Second moment of the infected population.
        SI (float): Susceptible-Infected interaction term.
        S2 (float): Second moment of the susceptible population.
        
        Returns:
        tuple: Derivatives (dS/dt, dI/dt, dSI/dt, dS2/dt, dI2/dt).
        """
        DSIDT = self.k1 * ((S2 * SI ** 2) / (I * S ** 2) - ((SI ** 2) * I2) / (S * I ** 2) - SI) - self.k2 * SI
        DS2DT = self.k1 * (SI - 2 * ((SI ** 2) * S2) / (S ** 2 * I))
        DI2DT = self.k1 * (2 * ((SI ** 2) * I2) / (S * I ** 2) + SI) + self.k2 * (I - 2 * I2)

        dSdt = -self.k1 * SI
        dIdt = self.k1 * SI - self.k2 * I

        return dSdt, dIdt, DSIDT, DS2DT, DI2DT
    
    def RK4_step(self, S, I, I2, SI, S2, dt):
        """
        Perform one step of the Runge-Kutta 4th order (RK4) method.
        
        Parameters:
        S (float): Susceptible population.
        I (float): Infected population.
        I2 (float): Second moment of the infected population.
        SI (float): Susceptible-Infected interaction term.
        S2 (float): Second moment of the susceptible population.
        dt (float): Time step for the simulation.
        
        Returns:
        tuple: Updated values (S_next, I_next, SI_next, S2_next, I2_next).
        """
        k1_S, k1_I, k1_SI, k1_S2, k1_I2 = self.differential(S, I, I2, SI, S2)
        k2_S, k2_I, k2_SI, k2_S2, k2_I2 = self.differential(S + k1_S * dt / 2, I + k1_I * dt / 2, I2 + k1_I2 * dt / 2, SI + k1_SI * dt / 2, S2 + k1_S2 * dt / 2)
        k3_S, k3_I, k3_SI, k3_S2, k3_I2 = self.differential(S + k2_S * dt / 2, I + k2_I * dt / 2, I2 + k2_I2 * dt / 2, SI + k2_SI * dt / 2, S2 + k2_S2 * dt / 2)
        k4_S, k4_I, k4_SI, k4_S2, k4_I2 = self.differential(S + k3_S * dt, I + k3_I * dt, I2 + k3_I2 * dt, SI + k3_SI * dt, S2 + k3_S2 * dt)
        
        S_next = S + dt * (k1_S + 2 * k2_S + 2 * k3_S + k4_S) / 6
        I_next = I + dt * (k1_I + 2 * k2_I + 2 * k3_I + k4_I) / 6
        SI_next = SI + dt * (k1_SI + 2 * k2_SI + 2 * k3_SI + k4_SI) / 6
        S2_next = S2 + dt * (k1_S2 + 2 * k2_S2 + 2 * k3_S2 + k4_S2) / 6
        I2_next = I2 + dt * (k1_I2 + 2 * k2_I2 + 2 * k3_I2 + k4_I2) / 6

        return S_next, I_next, SI_next, S2_next, I2_next
    
    def run_deterministic_simulation(self):
        """
        Run the deterministic solver over the total number of time points.
        
        Returns:
        tuple: Time series of S and I.
        """
        S = np.zeros(self.num_points)  # Susceptible vector
        I = np.zeros(self.num_points)  # Infected vector
        
        S2 = np.zeros(self.num_points)  # Second moment of susceptible vector
        I2 = np.zeros(self.num_points)  # Second moment of infected vector
        SI = np.zeros(self.num_points)  # Susceptible-Infected interaction term vector

        S[0] = self.S0  # Initial susceptible population
        I[0] = self.I0  # Initial infected population
        SI[0] = self.SI0
        S2[0] = self.S02
        I2[0] = self.I02
        
        # Main loop over the time steps
        for i in range(1, self.num_points):
            S[i], I[i], SI[i], S2[i], I2[i] = self.RK4_step(S[i - 1], I[i - 1], I2[i-1], SI[i-1], S2[i-1], self.dt)
        
        return S, I
    
    def run_combined(self, total_simulations):
        """
        Run the combined deterministic and stochastic models.
        
        Parameters:
        total_simulations (int): Number of stochastic simulations to run.
        
        Returns:
        tuple: Time series of S, I from deterministic model, and averaged states from stochastic model.
        """
        S, I = self.run_deterministic_simulation()
        data_table_cum = self.run_gillespie_simulation(total_simulations=total_simulations)
        S_stochastic  = data_table_cum[:, 0]
        I_stochastic = data_table_cum[:,1]

        return S, I, S_stochastic, I_stochastic
