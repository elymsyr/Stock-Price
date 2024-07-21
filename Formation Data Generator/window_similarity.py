import numpy as np
import matplotlib.pyplot as plt
import fastdtw
from scipy.spatial.distance import euclidean

def plot_patterns(main_array, found_patterns, patterns):
    """
    Plot the main array and highlight the found patterns with their original values.
    
    Parameters:
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
            # pattern_to_plot = main_array[idx:idx + pattern_length]
            pattern_to_plot = patterns[pattern_index]
            plt.plot(range(idx, idx + pattern_length), pattern_to_plot, 
                     label=f'Pattern {pattern_index+1}', color=colors[pattern_index % len(colors)])
    
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Detected Patterns in Main Array')
    plt.show()

def mse(array1, array2):
    """Compute Mean Squared Error between two arrays"""
    return np.mean((array1 - array2) ** 2)

def normalize(array):
    """Normalize an array to the range [0, 1]"""
    min_val = np.min(array)
    max_val = np.max(array)
    if max_val - min_val == 0:
        return array - min_val  # Avoid division by zero if all values are the same
    return (array - min_val) / (max_val - min_val)

def find_patterns(main_array, *patterns, threshold=0.01):
    """
    Find similar patterns in the main array.
    
    Parameters:
    main_array (numpy.ndarray): The main array to search in.
    patterns (list of numpy.ndarray): The patterns to search for.
    threshold (float): The threshold for similarity.
    
    Returns:
    dict: A dictionary where keys are pattern indices and values are lists of starting indices in the main array.
    """
    main_array = main_array.flatten()
    results = {i: [] for i in range(len(patterns))}
    
    # Normalize the patterns
    normalized_patterns = [normalize(pattern.flatten()) for pattern in patterns]
    
    for pattern_index, pattern in enumerate(normalized_patterns):
        pattern_length = len(pattern)
        
        for i in range(len(main_array) - pattern_length + 1):
            window = main_array[i:i + pattern_length]
            normalized_window = normalize(window)
            if mse(normalized_window, pattern) <= threshold:
                results[pattern_index].append(i)
    
    return results


FORMATIONS = {
    'ascending_triangle': [0.9, 0.1, 0.9, 0.52, 0.1, 0.26, 0.1],
    'descending_triangle': [0.9, 0.1, 0.5, 0.1, 0.29, 0.1],
    'symmetrical_triangle': [0.9, 0.1, 0.7, 0.3, 0.4, 0.15, 0.2, 0.11, 0.1]
}

# Example usage
main_array = np.array([0.1, 0.12, 0.09, 0.18, 0.02, 0.1, 0.02, 0.58, 0.02, 1.0]).reshape(-1, 1)
pattern1 = np.array([0.98, 0.11, 0.49, 0.12]).reshape(-1, 1)
pattern2 = np.array([0.1, 0.26, 0.04]).reshape(-1, 1)

patterns = [pattern1, pattern2]

found_patterns = find_patterns(main_array, *patterns, threshold=0.01)
print("Found patterns:", found_patterns)

plot_patterns(main_array, found_patterns, patterns)