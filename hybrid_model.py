import numpy as np
import matplotlib.pyplot as plt
import random
import tqdm

"""Parameters"""
DS_0 = 400
DI_0 = 5
CS_0 = 0
CI_0 = 0
k1 = 0.002
k2 = 0.1
dt = 0.001
tf = 30
T1 = 600
gamma = 1
number_molecules = 4

num_points = int(tf / dt) + 1  # Number of time points in the simulation
timegrid = np.linspace(0, tf, num_points, dtype=np.float64)  # Time points
data_table_init = np.zeros((num_points, number_molecules), dtype=np.float64)  # Matrix to store simulation results
data_table_cum = np.zeros((num_points, number_molecules), dtype=np.float64) #The cumulative simulation results

combined = np.zeros((num_points, 1), dtype=np.float64)

#Note the vector is like this:
"""
Vector = 
[
DS
DI
CS
CI
]
"""
states = np.array([DS_0,DI_0,CS_0,CI_0], dtype=float)

S_matrix = np.array([[-1,1,0,0],
                     [0,1,-1,0],
                     [0,-1,0,0],
                     [1,0,-1,0],
                     [0,1,0,-1],
                     [0,-1,0,1]],dtype=float)

"""FORWARD IS FROM DISCRETE TO CONTINIOUS"""
def compute_propensities(states):

    DS,DI,CS,CI = states
    alpha_1 = k1*DS*DI
    alpha_2 = k1*CS*DI
    alpha_3 = k2*DI

    alpha_bS = gamma * CS if states.sum() <= T1 else 0# Continious S to discrete S

    alpha_bI = gamma * CI if states.sum()  <= T1 else 0 # Cont I to Discrete I

    alpha_fS = gamma * DS if states.sum() > T1 else 0# Discrete S to continous S
    
    alpha_fI = gamma * DI if states.sum() >T1 else 0 # Discrete I to Cont I
    

    return np.array([alpha_1,alpha_2,alpha_3,alpha_bS,alpha_bI,alpha_fS,alpha_fI])







