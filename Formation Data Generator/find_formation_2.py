import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw # type: ignore
from scipy.spatial.distance import euclidean

def scale_list(numbers, min_val = None, max_val = None):
    if not min_val: min_val = min(numbers)
    if not max_val: max_val = max(numbers)
    scaled_numbers = [(x - min_val) / (max_val - min_val) for x in numbers]
    return scaled_numbers

def find_patterns(main_array, *patterns, threshold=0.1, data_size = 15):
    """
    Find similar patterns in the main array using DTW.

    Parameters:
    main_array (numpy.ndarray): The main array to search in.
    patterns (list of numpy.ndarray): The patterns to search for.
    threshold (float): The threshold for similarity.

    Returns:
    dict: A dictionary where keys are pattern indices and values are lists of starting indices in the main array.
    """
    main_array = main_array.flatten()
    results = {i: [] for i in range(len(patterns))}

    for pattern_index, pattern in enumerate(patterns):
        for i in range(len(main_array) - data_size + 1):
            window = main_array[i:i + data_size]
            distance, _ = fastdtw(window.reshape(-1, 1), pattern.reshape(-1, 1), dist=euclidean)
            if distance <= threshold:
                results[pattern_index].append(i)

    return results

def plot_patterns(main_array, found_patterns, patterns):
    """
    Plot the main array and highlight the found patterns with their original values.

    Parameters
    main_array (numpy.ndarray): The main array.
    found_patterns (dict): The dictionary of found patterns.
    patterns (list of numpy.ndarray): The list of patterns.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(main_array, label='Main Array', color='blue')
    
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    
    for pattern_index, indices in found_patterns.items():
        pattern_length = len(patterns[pattern_index])
        for idx in indices:
            pattern_to_plot = main_array[idx:idx + pattern_length]
            plt.plot(range(idx, idx + pattern_length), pattern_to_plot, 
                     label=f'Pattern {pattern_index+1}', color=colors[pattern_index % len(colors)])
    
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Detected Patterns in Main Array')
    plt.show()

# Example usage
main_array = np.array([0.9, 0.1, 0.9, 0.1, 0.5, 0.1, 0.12, 0.12, 0.09, 0.18, 0.1, 0.12, 0.09, 0.18, 0.02, 0.1, 0.9, 0.45, 0.23, 0.1, 0.30, 0.6, 0.9, 0.6, 0.3, 0.1, 0.02, 0.58, 0.02, 1.0, 0.12, 0.09, 0.18, 0.02, 0.1,  0.02, 0.1, 0.02, 0.58]).reshape(-1, 1)
pattern1 = np.array([0.9, 0.1, 0.5, 0.1, 0.29, 0.1]).reshape(-1, 1)
pattern2 = np.array([0.9, 0.1, 0.9, 0.1]).reshape(-1, 1)

patterns = [pattern1, pattern2]

found_patterns = find_patterns(main_array, *patterns, threshold=1)
print("Found patterns:", found_patterns)

# Plotting the patterns
plot_patterns(main_array, found_patterns, patterns)
