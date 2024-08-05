import numpy as np

class ODE_kirkwood:
    """
    Class for simulating the SIR (Susceptible-Infected-Recovered) model using the Kirkwood approximation.
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
        self.SI = S0 * I0  # Initial value of the SI product
        self.S2 = S0 * S0  # Initial value of the S^2 term
        self.I2 = I0 * I0  # Initial value of the I^2 term

        self.epsilon = 1e-10  # Small constant to prevent division by zero

        self.num_points = int(tf / dt) + 1  # Number of time points
        self.timegrid = np.linspace(0, tf, self.num_points)  # Time grid

        # Arrays to store the results
        self.particles = np.zeros((self.num_points, 5))

        # Initialize the first time step with initial conditions
        self.particles[0, :] = np.array([self.S0, self.I0, self.SI, self.S2, self.I2])

    def differential_kirkwood(self, S, I, SI, S2, I2):
        """
        Computes the differentials for the Kirkwood approximation.

        Parameters:
        S (float): Proportion of susceptible individuals
        I (float): Proportion of infected individuals
        SI (float): Product of S and I
        S2 (float): Square of S
        I2 (float): Square of I

        Returns:
        dSdt (float): Time derivative of susceptible individuals
        dIdt (float): Time derivative of infected individuals
        DSIDT (float): Time derivative of SI
        DS2DT (float): Time derivative of S^2
        DI2DT (float): Time derivative of I^2
        """
        DSIDT = self.k1 * ((S2 * SI ** 2) / (I * S ** 2) - ((SI ** 2) * I2) / (S * I ** 2) - SI) - self.k2 * SI
        DS2DT = self.k1 * (SI - 2 * ((SI ** 2) * S2) / (S ** 2 * I))
        DI2DT = self.k1 * (2 * ((SI ** 2) * I2) / (S * I ** 2) + SI) + self.k2 * (I - 2 * I2)
        dSdt = -self.k1 * SI
        dIdt = self.k1 * SI - self.k2 * I
        return dSdt, dIdt, DSIDT, DS2DT, DI2DT

    def RK4_kirkwood(self, S, I, SI, S2, I2):
        """
        Performs a single Runge-Kutta 4th order (RK4) step for the Kirkwood approximation.

        Parameters:
        S (float): Proportion of susceptible individuals at the current time step
        I (float): Proportion of infected individuals at the current time step
        SI (float): Product of S and I at the current time step
        S2 (float): Square of S at the current time step
        I2 (float): Square of I at the current time step

        Returns:
        S_next (float): Proportion of susceptible individuals at the next time step
        I_next (float): Proportion of infected individuals at the next time step
        SI_next (float): Product of S and I at the next time step
        S2_next (float): Square of S at the next time step
        I2_next (float): Square of I at the next time step
        """
        k1_S, k1_I, k1_SI, k1_S2, k1_I2 = self.differential_kirkwood(S, I, SI, S2, I2)
        k2_S, k2_I, k2_SI, k2_S2, k2_I2 = self.differential_kirkwood(S + k1_S * self.dt / 2, I + k1_I * self.dt / 2, SI + k1_SI * self.dt / 2, S2 + k1_S2 * self.dt / 2, I2 + k1_I2 * self.dt / 2)
        k3_S, k3_I, k3_SI, k3_S2, k3_I2 = self.differential_kirkwood(S + k2_S * self.dt / 2, I + k2_I * self.dt / 2, SI + k2_SI * self.dt / 2, S2 + k2_S2 * self.dt / 2, I2 + k2_I2 * self.dt / 2)
        k4_S, k4_I, k4_SI, k4_S2, k4_I2 = self.differential_kirkwood(S + k3_S * self.dt, I + k3_I * self.dt, SI + k3_SI * self.dt, S2 + k3_S2 * self.dt, I2 + k3_I2 * self.dt)

        S_next = S + self.dt * (k1_S + 2 * k2_S + 2 * k3_S + k4_S) / 6
        I_next = I + self.dt * (k1_I + 2 * k2_I + 2 * k3_I + k4_I) / 6
        SI_next = SI + self.dt * (k1_SI + 2 * k2_SI + 2 * k3_SI + k4_SI) / 6
        S2_next = S2 + self.dt * (k1_S2 + 2 * k2_S2 + 2 * k3_S2 + k4_S2) / 6
        I2_next = I2 + self.dt * (k1_I2 + 2 * k2_I2 + 2 * k3_I2 + k4_I2) / 6

        return S_next, I_next, SI_next, S2_next, I2_next

    def run_kirkwood(self):
        """
        Runs the Kirkwood approximation simulation.

        Returns:
        S (numpy.ndarray): Array of susceptible proportions over time
        I (numpy.ndarray): Array of infected proportions over time
        SI (numpy.ndarray): Array of SI products over time
        S2 (numpy.ndarray): Array of S^2 terms over time
        I2 (numpy.ndarray): Array of I^2 terms over time
        """
        for i in range(1, self.num_points):
            self.particles[i, :] = self.RK4_kirkwood(self.particles[i-1, 0], self.particles[i-1, 1], self.particles[i-1, 2], self.particles[i-1, 3], self.particles[i-1, 4])
        return self.particles[:, 0], self.particles[:, 1], self.particles[:, 2], self.particles[:, 3], self.particles[:, 4]
