from functions import *

# Generate the original time series
data_points = np.random.uniform(100, 200, 20)
time_series = generate_time_series(data_points)

# Interpolate the time series to generate daily data points
interpolated_time_series = interpolate_time_series(time_series)

# # Plotting the interpolated time series data
# plt.figure(figsize=(10, 5))
# plt.plot(interpolated_time_series['Timestamp'], interpolated_time_series['Value'], label='Interpolated Data', linestyle='-', color='gray')
# plt.scatter(time_series['Timestamp'], time_series['Value'], color='black', label='Original Data Points', marker='o')
# plt.title('Interpolated Time Series with Original Data Points')
# plt.xlabel('Timestamp')
# plt.ylabel('Value')
# plt.legend()
# plt.grid(True)

data = interpolated_time_series['Value'].tolist()
time = list(range(1, len(interpolated_time_series) + 1))

for distance in [20,40,60,80,100,120,140,160,180,200]:
    maxima, minima = find_local_extrema(data, time, distance)
    
    merged_list = sorted(minima + maxima, key=lambda x: x[0])
    
    # Extracting indices and values for plotting
    indices = [item[0] for item in merged_list]
    values = [item[1] for item in merged_list]
    plt.figure(figsize=(10, 5))
    
    plt.plot(indices, values, marker='o')
    plt.plot(time, data, label='Data')
    plt.scatter(*zip(*maxima), color='red', label='Maxima')
    plt.scatter(*zip(*minima), color='blue', label='Minima')

    plt.title(f'Local Maxima and Minima with Distance {distance}')
    plt.xlabel('Time')
    plt.ylabel('Data Value')
    plt.legend()
    plt.grid(True)

    plt.show()





