import numpy as np
import matplotlib.pyplot as plt
import random
import tqdm
import time
class HybridModelLogistic:
    def __init__(self, DS_0=400, DI_0=5, CS_0=0, CI_0=0, k1=0.002, k2=0.1, dt=0.2, tf=40, threshold_centre_infected = 20, threshold_centre_susceptible = 50, gradient = 0.1, intensity = 1 , gamma=0.5):
        """
        Initialize the hybrid model with the given parameters.
        
        Parameters:
        DS_0 (int): Initial discrete susceptible population.
        DI_0 (int): Initial discrete infected population.
        CS_0 (int): Initial continuous susceptible population.
        CI_0 (int): Initial continuous infected population.
        k1 (float): Infection rate.
        k2 (float): Recovery rate.
        dt (float): Time step for the simulation.
        tf (float): Final time of the simulation.
        T1 (float): Threshold for continuous to discrete infected conversion.
        T2 (float): Threshold for continuous to discrete susceptible conversion.
        gamma (float): Conversion rate between discrete and continuous populations.
        """
        self.DS_0 = DS_0
        self.DI_0 = DI_0
        self.CS_0 = CS_0
        self.CI_0 = CI_0
        self.k1 = k1
        self.k2 = k2
        self.dt = dt
        self.tf = tf
        self.number_molecules = 2  # Number of molecules (Susceptible, Infected)


        self.threshold_centre_infected = threshold_centre_infected
        self.threshold_centre_susceptible = threshold_centre_susceptible



        self.gradient = gradient
        self.intensity = intensity

        self.num_points = int(tf / dt) + 1  # Number of time points in the simulation
        self.timegrid = np.linspace(0, tf, self.num_points, dtype=np.float64)  # Time points
        self.data_table_init = np.zeros((self.num_points, 2 * self.number_molecules), dtype=np.float64)  # Matrix to store simulation results
        self.data_table_cum = np.zeros((self.num_points, 2 * self.number_molecules), dtype=np.float64)  # Cumulative simulation results


        self.threshold_I_vector = np.ones((self.num_points,1),dtype = np.float64 )*threshold_centre_infected
        self.threshold_S_vector = np.ones((self.num_points,1),dtype = np.float64 )*threshold_centre_susceptible

        self.DS_vector = np.zeros((self.num_points,1),dtype = np.float64 )
        self.DI_vector = np.zeros((self.num_points,1),dtype = np.float64 )
        self.CS_vector = np.zeros((self.num_points,1),dtype = np.float64 )
        self.CI_vector = np.zeros((self.num_points,1),dtype = np.float64 )

        self.S_vector = np.zeros((self.num_points,1),dtype = np.float64 )
        self.I_vector = np.zeros((self.num_points,1),dtype = np.float64 )
       


        self.states_init = np.array([DS_0, DI_0, CS_0, CI_0], dtype=float)  # Initial states
        self.S_matrix = np.array([
            [-1, 1, 0, 0],
            [0, 1, -1, 0],
            [0, -1, 0, 0],
            [-1, 0, 0, 1],
            [1, 0, -1, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, -1, 0, 1]
        ], dtype=int)  # Stoichiometric matrix for reactions



    def logistic_function(self,DS,DI,CS,CI):

            """
            Logistic function for the propensity calculation
            Input: The gradient, Theeshold centre and intensity, Particle values
            Returns: A propensity Logistic function.
            """
            alpha_fI = DI*self.intensity/(1+np.exp(-self.gradient*(DI-self.threshold_centre_infected))) #From discrete to continious
            alpha_bI = CI*self.intensity/(1+np.exp(self.gradient*(CI-self.threshold_centre_infected))) #From continious back to discrete
            
            alpha_fS = DS*self.intensity/(1+np.exp(-self.gradient*(DS-self.threshold_centre_susceptible))) #From Discrete to continious
            alpha_bS = CS*self.intensity/(1+np.exp(self.gradient*(CS-self.threshold_centre_susceptible)))
            #From continious to discrete
            
            return alpha_fI, alpha_bI, alpha_fS, alpha_bS 
        

    def compute_propensities(self, states):
        """
        Compute the propensity functions for the reactions.
        
        Parameters:
        states (numpy array): Current states of the system.
        
        Returns:
        numpy array: Propensity functions [alpha1, alpha2, alpha3, alpha4, alpha_bS, alpha_bI, alpha_fS, alpha_fI].
        """
        DS, DI, CS, CI = states
        # Propensities for dynamics
        alpha_1 = self.k1 * DS * DI
        alpha_2 = self.k1 * CS * DI
        alpha_3 = self.k2 * DI
        alpha_4 = self.k1 * DS * CI
        # Propensities for conversion (using logistic function)
        alpha_fI, alpha_bI, alpha_fS, alpha_bS = self.logistic_function(DS,DI,CS,CI)

        return np.array([alpha_1, alpha_2, alpha_3, alpha_4, alpha_bS, alpha_bI, alpha_fS, alpha_fI])

    def perform_reaction(self, index, states):
        """
        Perform the selected reaction and update the states.
        
        Parameters:
        index (int): Index of the reaction to perform.
        states (numpy array): Current states of the system.
        
        Returns:
        numpy array: Updated states after the reaction.
        """
        _, _, CS, CI = states
        if index == 4 and CS < 1:
            if CS >= random.uniform(0, 1):
                states[0] += self.S_matrix[index][0]  # Update discrete molecules
                states[2] = 0  # Reset continuous molecules
        elif index == 5 and CI < 1:
            if CI >= random.uniform(0, 1):
                states[1] += self.S_matrix[index][1]  # Update discrete molecules
                states[3] = 0  # Reset continuous molecules
        else:
            states += self.S_matrix[index]  # General update for other reactions

        return states

    def gillespie_step(self, alpha_cum, alpha0, states):
        """
        Perform one step of the Gillespie algorithm.
        
        Parameters:
        alpha_cum (numpy array): Cumulative sum of the propensity functions.
        alpha0 (float): Sum of the propensity functions.
        states (numpy array): Current states of the system.
        
        Returns:
        numpy array: Updated states after the reaction.
        """
        r2 = random.uniform(0, 1)  # Generate random number for reaction selection
        index = next(i for i, alpha in enumerate(alpha_cum / alpha0) if r2 <= alpha)  # Determine which reaction occurs
        return self.perform_reaction(index, states)  # Update states based on selected reaction

    def update_ode(self, states):
        """
        Update the states using the ODE with RK4 method.
        
        Parameters:
        states (numpy array): Current states of the system.
        
        Returns:
        numpy array: Updated states after one step of the ODE is performed.
        """
        def differential(S, I):
            DsDt = -self.k1 * S * I
            DiDt = self.k1 * S * I - self.k2 * I
            return DsDt, DiDt

        def RK4(states):
            _, _, S, I = states
            P1 = differential(S, I)
            P2 = differential(S + P1[0] * self.dt / 2, I + P1[1] * self.dt / 2)
            P3 = differential(S + P2[0] * self.dt / 2, I + P2[1] * self.dt / 2)
            P4 = differential(S + P3[0] * self.dt, I + P3[1] * self.dt)
            return S + (P1[0] + 2 * P2[0] + 2 * P3[0] + P4[0]) * self.dt / 6, I + (P1[1] + 2 * P2[1] + 2 * P3[1] + P4[1]) * self.dt / 6

        rk4_result = RK4(states)
        states[2] = max(rk4_result[0], 0)
        states[3] = max(rk4_result[1], 0)
        return states



    # def update_ode(self, states):
    #     """
    #     Update the states using the ODE with RK4 method.
        
    #     Parameters:
    #     states (numpy array): Current states of the system.
        
    #     Returns:
    #     numpy array: Updated states after one step of the ODE is performed.
    #     """


    #     def differential(S, I, I2, SI, S2):
    #         DSIDT = self.k1 * ((S2 * SI ** 2) / (I * S ** 2) - ((SI ** 2) * I2) / (S * I ** 2) - SI) - self.k2 * SI
    #         DS2DT = self.k1 * (SI - 2 * ((SI ** 2) * S2) / (S ** 2 * I))
    #         DI2DT = self.k1 * (2 * ((SI ** 2) * I2) / (S * I ** 2) + SI) + self.k2 * (I - 2 * I2)

    #         dSdt = -self.k1 * SI
    #         dIdt = self.k1 * SI - self.k2 * I

    #         return dSdt, dIdt, DSIDT, DS2DT, DI2DT

    #     def RK4(states):
    #         _, _, S, I = states
    #         self.I2 = I ** 2
    #         self.SI = S * I
            
    #         k1_S, k1_I, k1_SI, k1_S2, k1_I2 = self.differential(S, I, self.I2, self.SI, self.S2)
    #         k2_S, k2_I, k2_SI, k2_S2, k2_I2 = self.differential(S + k1_S * self.dt / 2, I + k1_I * self.dt / 2, self.I2 + k1_I2 * self.dt / 2, self.SI + k1_SI * self.dt / 2, self.S2 + k1_S2 * self.dt / 2)
    #         k3_S, k3_I, k3_SI, k3_S2, k3_I2 = self.differential(S + k2_S * self.dt / 2, I + k2_I * self.dt / 2, self.I2 + k2_I2 * self.dt / 2, self.SI + k2_SI * self.dt / 2, self.S2 + k2_S2 * self.dt / 2)
    #         k4_S, k4_I, k4_SI, k4_S2, k4_I2 = self.differential(S + k3_S * self.dt, I + k3_I * self.dt, self.I2 + k3_I2 * self.dt, self.SI + k3_SI * self.dt, self.S2 + k3_S2 * self.dt)
        
    #         S_next = S + self.dt * (k1_S + 2 * k2_S + 2 * k3_S + k4_S) / 6
    #         I_next = I + self.dt * (k1_I + 2 * k2_I + 2 * k3_I + k4_I) / 6
    #         SI_next = self.SI + self.dt * (k1_SI + 2 * k2_SI + 2 * k3_SI + k4_SI) / 6
    #         S2_next = self.S2 + self.dt * (k1_S2 + 2 * k2_S2 + 2 * k3_S2 + k4_S2) / 6
    #         I2_next = self.I2 + self.dt * (k1_I2 + 2 * k2_I2 + 2 * k3_I2 + k4_I2) / 6

    #         return S_next, I_next, SI_next, S2_next, I2_next




    #     rk4_result = RK4(states)
    #     states[2] = max(rk4_result[0], 0)
    #     states[3] = max(rk4_result[1], 0)
    #     return states
    

    def run_simulation(self):
        """
        Run a single simulation of the hybrid model.
        
        Returns:
        numpy array: Data table of the states over time.
        """
        t = 0
        old_time = t
        td = self.dt
        data_table = np.zeros((self.num_points, 2 * self.number_molecules), dtype=np.float64)
        states = self.states_init.copy()
        data_table[0, :] = states

        while t < self.tf:
            alpha_list = self.compute_propensities(states)
            alpha0 = sum(alpha_list)
            alpha_cum = np.cumsum(alpha_list)

            if alpha0 >= 1e-10:
                tau = np.log(1 / random.uniform(0, 1)) / alpha0

                if t + tau <= td:
                    states = self.gillespie_step(alpha_cum, alpha0, states)
                    old_time = t
                    t += tau

                    ind_before = np.searchsorted(self.timegrid, old_time, 'right')
                    ind_after = np.searchsorted(self.timegrid, t, 'left')
                    for index in range(ind_before, min(ind_after + 1, self.num_points)):
                        data_table[index, :] = states
                else:
                    states = self.update_ode(states)
                    t = td
                    td += self.dt

                    index = min(np.searchsorted(self.timegrid, t + 1e-10, 'left'), self.num_points - 1)
                    data_table[index, :] = states
            else:
                states = self.update_ode(states)
                t = td
                td += self.dt

                index = min(np.searchsorted(self.timegrid, t + 1e-10, 'right'), self.num_points - 1)
                data_table[index, :] = states

        return data_table

    def run_multiple(self, total_simulations=100):
        """
        Run the simulation multiple times and compute the average results.
        
        Parameters:
        total_simulations (int): Number of simulations to run.
        
        Returns:
        tuple: Time grid, Chemical vectors DS,DI,CS,CI and combined 
        """
        #DS,DI,CS,CI
        print("Running multiple simulations of the Hybrid_model...")

        start_time = time.perf_counter()
        for _ in tqdm.tqdm(range(total_simulations)):
            self.data_table_cum += self.run_simulation()
        
        self.total_time = time.perf_counter() - start_time
        self.data_table_cum /= total_simulations  # Average the results
        
        self.DS_vector = self.data_table_cum[:,0]
        self.DI_vector = self.data_table_cum[:,1]
        self.CS_vector = self.data_table_cum[:,2]
        self.CI_vector = self.data_table_cum[:,3]

        self.S_vector = self.DS_vector + self.CS_vector
        self.I_vector = self.DI_vector + self.CI_vector

        return self.timegrid, self.DS_vector, self.DI_vector, self.CS_vector, self.CI_vector, self.S_vector, self.I_vector
