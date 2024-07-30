import numpy as np
import matplotlib.pyplot as plt
import random
import tqdm


class HybridModel:

    def __init__(self,DS_0 = 400, DI_0 = 5, CS_0 = 0,CI_0 = 0, k1=0.002, k2=0.1, dt=0.2, tf=40, T1=40, T2=40, gamma=0.5):

        print("Setting up the class function, please run the  'Run multiple' function with total simulations")
        print("This will output, ")
        self.DS_0 = DS_0
        self.DI_0 = DI_0
        self.CS_0 = CS_0
        self.CI_0 = CI_0
        self.k1 = k1
        self.k2 = k2
        self.dt = dt
        self.tf = tf
        self.T1 = T1
        self.T2 = T2
        self.gamma = gamma
        self.number_molecules = 2

        self.num_points = int(tf / dt) + 1  # Number of time points in the simulation
        self.timegrid = np.linspace(0, tf, self.num_points, dtype=np.float64)  # Time points
        self.data_table_init = np.zeros((self.num_points, 2*self.number_molecules), dtype=np.float64)  # Matrix to store simulation results
        self.data_table_cum = np.zeros((self.num_points, 2*self.number_molecules), dtype=np.float64) #The cumulative simulation results

        self.combined = np.zeros((self.num_points, self.number_molecules), dtype=np.float64) #Combined totals 

        self.states_init = np.array([DS_0, DI_0, CS_0, CI_0], dtype=float)
        self.S_matrix = np.array([
            [-1, 1, 0, 0],
            [0, 1, -1, 0],
            [0, -1, 0, 0],
            [-1, 0, 0, 1],
            [1, 0, -1, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, -1, 0, 1]
        ], dtype=int)


    def compute_propensities(self, states):
        """This will return the 8 propensity functions involved in the 8 different reactions"""
        DS, DI, CS, CI = states
        """First are the propensities involved in the dynamics"""
        alpha_1 = self.k1 * DS * DI
        alpha_2 = self.k1 * CS * DI
        alpha_3 = self.k2 * DI
        alpha_4 = self.k1 * DS * CI
        """The next are the propensities involved in conversion"""
        alpha_bS = self.gamma * CS if CS+DS < self.T2 else 0 # Continious S to discrete S
        alpha_bI = self.gamma * CI if CI+DI  < self.T1 else 0 # Cont I to Discrete I
        alpha_fS = self.gamma * DS if  CS+DS >= self.T2 else 0 # Discrete S to continous S
        alpha_fI = self.gamma * DI if CI+DI >= self.T1 else 0 # Discrete I to Cont I
    

        return np.array([alpha_1, alpha_2, alpha_3, alpha_4, alpha_bS, alpha_bI, alpha_fS, alpha_fI])
    

    def perform_reaction(self,index,states):
        """THis performs the reaction, it update the states depending on which reaction is selected (based off random number)"""
        _,_,CS,CI = states
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
    
    def gillespie_step(self,alpha_cum, alpha0, states):
        r2 = random.uniform(0, 1)  # Generate random number for reaction selection
        index = next(i for i, alpha in enumerate(alpha_cum / alpha0) if r2 <= alpha)  # Determine which reaction occurs
        return self.perform_reaction(index, states)  # Update states based on selected reaction


    
    def update_ode(self,states):
        def differential(S,I): 
            DsDt = -self.k1*S*I
            DiDt = self.k1*S*I-self.k2*I
            return DsDt, DiDt

        def RK4(states):
            _,_,S,I = states
            P1 = differential(S,I)
            P2 = differential(S+P1[0]*self.dt/2,I+P1[1]*self.dt/2)
            P3 = differential(S+P2[0]*self.dt/2,I+P2[1]*self.dt/2)
            P4 = differential(S+P3[0]*self.dt,I+P3[1]*self.dt)
            return S + (P1[0]+2*P2[0]+2*P3[0]+P4[0])*self.dt/6, I + (P1[1]+2*P2[1]+2*P3[1]+P4[1])*self.dt/6
        
        rk4_result = RK4(states)
        states[2] = max(rk4_result[0], 0)
        states[3] = max(rk4_result[1], 0)
        # Debug print statement
        #print(f"ODE Update: CS={states[2]}, CI={states[3]}")
        return states


    def run_simulation(self):
        t = 0
        old_time = t
        td = self.dt
        data_table = np.zeros((self.num_points, 2*self.number_molecules), dtype=np.float64)
        
        states = self.states_init.copy()
        data_table[0,:] = states

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
                    #print("alpha =0  , running ODE")
                    states = self.update_ode(states)
                    t = td
                    td += self.dt

                    index = min(np.searchsorted(self.timegrid, t + 1e-10, 'left'), self.num_points - 1)
                    data_table[index, :] = states
            else:
                states = self.update_ode(states)
                t = td
                td += self.dt

                index = min(np.searchsorted(self.timegrid, t + 1e-10, 'right'), self.num_points -1)
                data_table[index, :] = states

        return data_table
    
    def run_multiple(self,total_simulations = 100):
        """Run the simulation multiple times"""
        for i in tqdm.tqdm(range(total_simulations)):
            self.data_table_cum += self.run_simulation()
        """Output average"""
        self.data_table_cum /= total_simulations 
        self.combined[:, 0] = self.data_table_cum[:, 0] + self.data_table_cum[:, 2]
        self.combined[:, 1] = self.data_table_cum[:, 1] + self.data_table_cum[:, 3]


        return self.timegrid,self.data_table_cum,self.combined

