import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats

# Load data

# DS_0 = 200
# CS_0 = 0
# CI_0 = 0
# k1 = 0.002
# k2 = 0.1
# dt = 0.2
# tf = 60
# T1 = 25
# T2 = T1
# gamma = 2

# total_sims = 10
# discrete_infected_vector = np.arange(1, 10, 1)

# # Initialize matrices for errors
# repeats = 500



loaded_data = np.load('data.npz')
error_logistic_matrix = loaded_data['data1']
error_threshold_matrix = loaded_data['data2']


print(error_logistic_matrix.shape)
# Calculate average errors
avg_error_logistic = np.mean(error_logistic_matrix, axis=1)
avg_error_threshold = np.mean(error_threshold_matrix, axis=1)

# Plot the box plots and average errors
fig, axs = plt.subplots(1, 3, figsize=(20, 6))

# Plot the box plot for the logistic model
axs[0].boxplot(error_logistic_matrix.T)
axs[0].set_xlabel("Initial number of discrete Infected individuals")
axs[0].set_ylabel("Mean absolute error")
axs[0].set_title("Box plot of mean absolute error for the logistic model")

# Plot the box plot for the threshold model
axs[1].boxplot(error_threshold_matrix.T)
axs[1].set_xlabel("Initial number of discrete Infected individuals")
axs[1].set_ylabel("Mean absolute error")
axs[1].set_title("Box plot of mean absolute error for the threshold model")

# Plot the average errors
axs[2].plot(range(1, len(avg_error_logistic) + 1), avg_error_logistic, label='Logistic Model', marker='o')
axs[2].plot(range(1, len(avg_error_threshold) + 1), avg_error_threshold, label='Threshold Model', marker='o')
axs[2].set_xlabel("Initial number of discrete Infected individuals")
axs[2].set_ylabel("Average mean absolute error")
axs[2].set_title("Average mean absolute error for both models")
axs[2].legend()

# Display the plots
plt.tight_layout()
plt.show()

# Perform t-tests for each initial number of discrete infected individuals
for i in range(error_logistic_matrix.shape[0]):
    t_stat, p_value = scipy.stats.ttest_ind(error_logistic_matrix[i], error_threshold_matrix[i])
    print(f"Discrete Infected: {i+1}")
    print(f"T-test p-value: {p_value}")
    print("Statistically significant" if p_value < 0.05 else "Not statistically significant")
    print()