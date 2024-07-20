import numpy as np
import matplotlib.pyplot as plt

def generate_stock_like_data(data_points, num_points_per_segment, randomness=0.05):
    generated_data = []

    for i in range(len(data_points) - 1):
        start = data_points[i]
        end = data_points[i + 1]
        
        # Generate points between start and end
        segment_points = np.linspace(start, end, num_points_per_segment + 2)[1:-1]
        
        # Add randomness to generated points
        segment_points += np.random.uniform(-randomness, randomness, size=segment_points.shape)
        
        # Add the current data point
        if i == 0:
            generated_data.append(start)
        
        generated_data.extend(segment_points)
        
        # Add the next data point
        generated_data.append(end)

    return generated_data

# Example usage
data_points = [0.1, 0.6, 0.1, 0.6]
num_points_per_segment = 100
randomness = 0.05  # Adjust this value for more or less randomness
generated_list = generate_stock_like_data(data_points, num_points_per_segment, randomness)

# Generate x values for plotting
x_values = np.arange(len(generated_list))

# Generate x values for the original data points (for drawing lines)
original_x_values = np.linspace(0, len(generated_list) - 1, len(data_points))

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(x_values, generated_list, marker='o', linestyle='-', color='b', label='Generated Data')

# Draw lines between original data points
for i in range(len(data_points) - 1):
    plt.plot(
        [original_x_values[i], original_x_values[i + 1]], 
        [data_points[i], data_points[i + 1]], 
        linestyle='--', color='r', label='Original Data Points' if i == 0 else ''
    )

# Draw horizontal lines at each original data point
for i, value in enumerate(data_points):
    plt.axhline(y=value, linestyle=':', color='g', label='Horizontal Lines' if i == 0 else '')

plt.title('Generated Data with Random Variations, Lines Connecting Original Data Points, and Horizontal Lines')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
