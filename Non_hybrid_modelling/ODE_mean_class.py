import numpy as np

class ODE_mean_class:
    """
    Class for simulating the SIR (Susceptible-Infected-Recovered) model using the mean field approximation.
    """

    def __init__(self, S0, I0, k1, k2, dt, tf):
        """
        Initializes the ODE_class with the given parameters.

        Parameters:
        S0 (float): Initial proportion of susceptible individuals
        I0 (float): Initial proportion of infected individuals
        k1 (float): Infection rate constant
        k2 (float): Recovery rate constant
        dt (float): Time step for the simulation
        tf (float): Final time for the simulation
        """
        self.k1 = k1  # Infection rate constant
        self.k2 = k2  # Recovery rate constant
        self.dt = dt  # Time step
        self.tf = tf  # Final time
        self.S0 = S0  # Initial proportion of susceptible individuals
        self.I0 = I0  # Initial proportion of infected individuals

        self.num_points = int(tf / dt) + 1  # Number of time points
        self.timegrid = np.linspace(0, tf, self.num_points)  # Time grid

        # Arrays to store the results
        self.S = np.zeros(self.num_points)
        self.I = np.zeros(self.num_points)

        # Initialize the first time step with initial conditions
        self.S[0] = self.S0
        self.I[0] = self.I0

    def differential_mean(self, S, I):
        """
        Computes the differentials for the mean field model.

        Parameters:
        S (float): Proportion of susceptible individuals
        I (float): Proportion of infected individuals

        Returns:
        dSdt (float): Time derivative of susceptible individuals
        dIdt (float): Time derivative of infected individuals
        """
        dSdt = -self.k1 * S * I
        dIdt = self.k1 * S * I - self.k2 * I
        return dSdt, dIdt

    def RK4_step_mean(self, S, I):
        """
        Performs a single Runge-Kutta 4th order (RK4) step for the mean field model.

        Parameters:
        S (float): Proportion of susceptible individuals at the current time step
        I (float): Proportion of infected individuals at the current time step

        Returns:
        S_next (float): Proportion of susceptible individuals at the next time step
        I_next (float): Proportion of infected individuals at the next time step
        """
        k1_S, k1_I = self.differential_mean(S, I)
        k2_S, k2_I = self.differential_mean(S + k1_S * self.dt / 2, I + k1_I * self.dt / 2)
        k3_S, k3_I = self.differential_mean(S + k2_S * self.dt / 2, I + k2_I * self.dt / 2)
        k4_S, k4_I = self.differential_mean(S + k3_S * self.dt, I + k3_I * self.dt)
        S_next = S + self.dt * (k1_S + 2 * k2_S + 2 * k3_S + k4_S) / 6
        I_next = I + self.dt * (k1_I + 2 * k2_I + 2 * k3_I + k4_I) / 6
        return S_next, I_next

    def run_mean_field(self):
        """
        Runs the mean field simulation.

        Returns:
        S (numpy.ndarray): Array of susceptible proportions over time
        I (numpy.ndarray): Array of infected proportions over time
        """
        for i in range(1, self.num_points):
            self.S[i], self.I[i] = self.RK4_step_mean(self.S[i-1], self.I[i-1])
        return self.S, self.I
