import numpy as np
import matplotlib.pyplot as plt
import random
import tqdm

class StochasticSIR:
    """
    Class for simulating the SIR model using the Gillespie algorithm for stochastic simulations.
    """

    def __init__(self, S, I, k1, k2, tf, dt, number_molecules, total_simulations):
        """
        Initializes the StochasticSIR model with the given parameters.

        Parameters:
        S (int): Initial number of susceptible individuals
        I (int): Initial number of infected individuals
        k1 (float): Infection rate constant
        k2 (float): Recovery rate constant
        tf (float): Final time for the simulation
        dt (float): Time step for the simulation
        number_molecules (int): Number of molecules (states) in the simulation
        total_simulations (int): Number of simulations to run
        """
        self.S = S
        self.I = I
        self.k1 = k1
        self.k2 = k2
        self.tf = tf
        self.dt = dt
        self.number_molecules = number_molecules
        self.total_simulations = total_simulations

        self.num_points = int(tf / dt) + 1  # Number of time points in the simulation
        self.timegrid = np.linspace(0, tf, self.num_points, dtype=np.float64)  # Time points
        self.data_table_cum = np.zeros((self.num_points, number_molecules), dtype=np.float64)  # Cumulative simulation results
        
        self.states_init = np.array([S, I], dtype=int)  # States vector
    
        # Stoichiometric matrix for reactions
        self.S_matrix = np.array([[-1, 1],
                                  [0, -1]], dtype=int)

    def compute_propensities(self, states):
        """
        Computes the propensity functions for the given states.

        Parameters:
        states (numpy.ndarray): The current state vector [S, I]

        Returns:
        numpy.ndarray: Propensities for each reaction
        """
        S, I = states
        alpha_1 = self.k1 * S * I
        alpha_2 = self.k2 * I
        return np.array([alpha_1, alpha_2], dtype=float)

    def perform_reaction(self, index, states):
        """
        Updates the state vector based on the reaction index.

        Parameters:
        index (int): Index of the reaction that occurred
        states (numpy.ndarray): The current state vector [S, I]

        Returns:
        numpy.ndarray: Updated state vector
        """
        states += self.S_matrix[index]
        states[0] = np.max(states[0], 0)
        states[1] = np.max(states[1], 0)
        return states

    def gillespie_step(self, alpha_cum, alpha0, states):
        """
        Performs one step of the Gillespie algorithm.

        Parameters:
        alpha_cum (numpy.ndarray): Cumulative sum of propensities
        alpha0 (float): Sum of all propensities
        states (numpy.ndarray): The current state vector [S, I]

        Returns:
        numpy.ndarray: Updated state vector
        """
        r2 = random.uniform(0, 1)  # Generate random number for reaction selection
        index = next(i for i, alpha in enumerate(alpha_cum / alpha0) if r2 <= alpha)
        return self.perform_reaction(index, states)

    def run_simulation(self):
        """
        Runs the simulation for one realization and returns the results.

        Returns:
        numpy.ndarray: The data table with concentrations of susceptible and infected individuals over time
        """
        t = 0
        old_time = t
        data_table = np.zeros((self.num_points, self.number_molecules), dtype=np.float64)
        data_table[0,:] = self.states_init
        states = self.states_init.copy()

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

        return data_table

    def run_all_simulations(self):
        """
        Runs all simulations and calculates the average results.

        Returns:
        numpy.ndarray: The average data table with concentrations of susceptible and infected individuals over time
        """
        for _ in tqdm.tqdm(range(self.total_simulations)):
            self.data_table_cum += self.run_simulation()
        
        # Normalize by the number of simulations
        self.data_table_cum /= self.total_simulations
        return self.data_table_cum

