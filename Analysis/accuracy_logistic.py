import numpy as np
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Modelling import HybridModel, SISimulation_Mean, HybridModelLogistic # type: ignore

# Parameters for the simulation
DS_0 = 200
CS_0 = 0
CI_0 = 0
k1 = 0.002
k2 = 0.1
dt = 0.2
tf = 60
T1 = 25
T2 = T1
gamma = 2

total_sims = 10
discrete_infected_vector = np.arange(1, 10, 1)

# Initialize matrices for errors
repeats = 500

error_logistic_matrix = np.zeros((len(discrete_infected_vector), repeats))
error_threshold_matrix = np.zeros((len(discrete_infected_vector), repeats))

for i in range(len(discrete_infected_vector)):
    for j in range(repeats):
        print(f"Repeat number: {j}")
        DI_0 = discrete_infected_vector[i]

        combined_model = SISimulation_Mean(S0=DS_0, I0=DI_0, k1=k1, k2=k2, tf=tf, dt=dt)
        S_ODE, I_ODE, S_stochastic, I_stochastic = combined_model.run_combined(total_simulations=total_sims)

        hybrid_model = HybridModel(
            DS_0=DS_0, DI_0=DI_0, CS_0=CS_0, CI_0=CI_0,
            k1=k1, k2=k2, dt=dt, tf=tf, T1=T1, T2=T2, gamma=gamma
        )
        hybrid_model_logistic = HybridModelLogistic(
            DS_0=DS_0, DI_0=DI_0, CS_0=CS_0, CI_0=CI_0,
            k1=k1, k2=k2, dt=dt, tf=tf, threshold_centre_infected=T1, 
            threshold_centre_susceptible=T1, gradient=1, intensity=gamma, gamma=gamma
        )

        # Run models
        _, _, _, _, _, _, HI_vector = hybrid_model.run_multiple(total_simulations=total_sims)
        _, _, _, _, _, _, LogisticI_vector = hybrid_model_logistic.run_multiple(total_simulations=total_sims)

        hybrid_error_vector = np.abs(HI_vector - I_stochastic)
        logistic_error = np.abs(LogisticI_vector - I_stochastic)

        error_logistic_matrix[i, j] = np.mean(logistic_error)
        error_threshold_matrix[i, j] = np.mean(hybrid_error_vector)

# Create the /data directory if it doesn't exist

subdirectory = "data"

# Ensure the directory exists
os.makedirs(subdirectory, exist_ok=True)

# Define the file path within the subdirectory
file_path = os.path.join(subdirectory, "data.npz")

np.savez(file_path,data1 = error_logistic_matrix, data2 = error_threshold_matrix)


# fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# sns.boxplot(data=error_logistic_matrix.T, color="blue", ax=axes[0])
# axes[0].set_xlabel('Discrete Infected Vector', fontsize=14)
# axes[0].set_ylabel('Error', fontsize=14)
# axes[0].set_title('Logistic Error Distribution', fontsize=16)
# axes[0].set_ylim(0, max(np.max(error_logistic_matrix), np.max(error_threshold_matrix)))

# sns.boxplot(data=error_threshold_matrix.T, color="green", ax=axes[1])
# axes[1].set_xlabel('Discrete Infected Vector', fontsize=14)
# axes[1].set_ylabel('Error', fontsize=14)
# axes[1].set_title('Threshold Error Distribution', fontsize=16)
# axes[1].set_ylim(0, max(np.max(error_logistic_matrix), np.max(error_threshold_matrix)))

# # Annotate statistical significance
# pairs = [(i, i) for i in range(len(discrete_infected_vector))]
# annotator = Annotator(axes[0], pairs, data=error_logistic_matrix.T)
# annotator.configure(test='t-test_ind', text_format='star', loc='inside')
# annotator.apply_and_annotate()

# annotator = Annotator(axes[1], pairs, data=error_threshold_matrix.T)
# annotator.configure(test='t-test_ind', text_format='star', loc='inside')
# annotator.apply_and_annotate()

# plt.tight_layout()
# plt.show()