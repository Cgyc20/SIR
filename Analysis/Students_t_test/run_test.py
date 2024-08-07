import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

def load_data(error_file, threshold_file):
    try:
        error_matrix = np.load(error_file)
        threshold_vector = np.load(threshold_file)
    except IOError as e:
        print(f"Error loading files: {e}")
        return None, None
    return error_matrix, threshold_vector

def compute_statistics(error_matrix):
    mean_error = np.mean(error_matrix, axis=1)
    std_error = np.std(error_matrix, axis=1)
    return mean_error, std_error

def perform_t_tests(error_matrix, threshold_vector):
    comparison_vector = error_matrix[-1, :]
    for i in range(error_matrix.shape[0] ):
        print(f"Threshold value of {threshold_vector[i]}")
        t_statistic, p_value = ttest_ind(comparison_vector, error_matrix[i, :], equal_var=False)
        print(f"t-statistic: {t_statistic:.4f}")
        print(f"p-value: {p_value:.4f}")
        if p_value <= 0.1:
            print("Statistical significance: There is a significant difference.")
        else:
            print("Statistically insignificant!")

def plot_statistics(threshold_vector, mean_error, std_error):
    plt.figure(figsize=(10, 6))
    plt.plot(threshold_vector, mean_error, marker='o', color='black', linestyle=':', markersize=8, markerfacecolor='blue', label='Mean Error')
    plt.fill_between(threshold_vector, mean_error - std_error, mean_error + std_error, color='blue', alpha=0.2, label='Standard Deviation')
    plt.xlabel('Threshold')
    plt.ylabel('Error')
    plt.title('Mean and Standard Deviation of Error vs Threshold')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    error_matrix, threshold_vector = load_data('Error_matrix.npy', 'Threshold_vector.npy')
    if error_matrix is None or threshold_vector is None:
        return
    mean_error, std_error = compute_statistics(error_matrix)
    perform_t_tests(error_matrix, threshold_vector)
    plot_statistics(threshold_vector, mean_error, std_error)

if __name__ == "__main__":
    main()
