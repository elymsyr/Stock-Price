from scipy.signal import find_peaks

def find_local_extrema(data, time, distance):
    # Find indices of local maxima
    maxima_indices, _ = find_peaks(data, distance=distance)
    # Find indices of local minima by inverting the data
    minima_indices, _ = find_peaks([-x for x in data], distance=distance)
    
    maxima = [(time[i], data[i]) for i in maxima_indices]
    minima = [(time[i], data[i]) for i in minima_indices]
    
    # Scopes are defined as the difference in time between neighboring extrema
    scopes = []
    for i in range(1, len(maxima_indices)):
        scopes.append(time[maxima_indices[i]] - time[maxima_indices[i-1]])
    for i in range(1, len(minima_indices)):
        scopes.append(time[minima_indices[i]] - time[minima_indices[i-1]])
    
    return maxima, minima, scopes

# # Example usage
# data = interpolated_time_series['Value'].tolist()
# time = list(range(1, len(interpolated_time_series) + 1))
# maxima, minima, scopes = find_local_extrema(data, time, 50)

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(time, data, label='Data')
# plt.scatter(*zip(*maxima), color='red', label='Maxima')
# plt.scatter(*zip(*minima), color='blue', label='Minima')


# plt.title('Local Maxima and Minima')
# plt.xlabel('Time')
# plt.ylabel('Data Value')
# plt.legend()
# plt.grid(True)
# plt.show()