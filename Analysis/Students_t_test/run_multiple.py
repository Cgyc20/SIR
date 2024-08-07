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

def plot_statistics(threshold_vector, mean_errors, std_errors, suffixes):
    plt.figure(figsize=(10, 6))
    for mean_error, std_error, suffix in zip(mean_errors, std_errors, suffixes):
        plt.plot(threshold_vector, mean_error, marker='o', linestyle=':', markersize=8, label=f'Mean Error {suffix}')
        #plt.fill_between(threshold_vector, mean_error - std_error, mean_error + std_error, alpha=0.2, label=f'Standard Deviation {suffix}')
    plt.xlabel('Threshold')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    #suffixes = [100, 200, 300, 400]
    suffixes = [200, 400]
    mean_errors = []
    std_errors = []
    threshold_vector = None

    for suffix in suffixes:
        error_matrix, threshold_vector = load_data(f'Error_matrix_{suffix}.npy', f'Threshold_vector_{suffix}.npy')
        if error_matrix is None or threshold_vector is None:
            return
        mean_error, std_error = compute_statistics(error_matrix)
        mean_errors.append(mean_error)
        std_errors.append(std_error)

    plot_statistics(threshold_vector, mean_errors, std_errors, suffixes)

if __name__ == "__main__":
    main()