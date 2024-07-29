from functions import *

# Generate the original time series
data_points = np.random.uniform(100, 200, 20)
time_series = generate_time_series(data_points)

# Interpolate the time series to generate daily data points
interpolated_time_series = interpolate_time_series(time_series)

# # Plotting the interpolated time series data
# plt.figure(figsize=(10, 5))
# plt.plot(interpolated_time_series['Timestamp'], interpolated_time_series['Close'], label='Interpolated Data', linestyle='-', color='gray')
# plt.scatter(time_series['Timestamp'], time_series['Close'], color='black', label='Original Data Points', marker='o')
# plt.title('Interpolated Time Series with Original Data Points')
# plt.xlabel('Timestamp')
# plt.ylabel('Close')
# plt.legend()
# plt.grid(True)

data = interpolated_time_series['close'].tolist()
time = list(range(1, len(interpolated_time_series) + 1))
df = interpolated_time_series
distances = [40,60,80,100,120,140,160,180,200]

plt.figure(figsize=(30, 25))

for i, distance in enumerate(distances):
    # maxima, minima = find_local_extrema(data, time, distance)
    
    # merged_list = sorted(minima + maxima, key=lambda x: x[0])
    
    # # Extracting indices and values for plotting
    # indices = [item[0] for item in merged_list]
    # values = [item[1] for item in merged_list]
    # plt.figure(figsize=(10, 5))
    
    # plt.plot(indices, values, marker='o')
    # plt.plot(time, data, label='Data')
    # plt.scatter(*zip(*maxima), color='red', label='Maxima')
    # plt.scatter(*zip(*minima), color='blue', label='Minima')

    # plt.title(f'Local Maxima and Minima with Distance {distance}')
    # plt.xlabel('Time')
    # plt.ylabel('Data Value')
    # plt.legend()
    
    plt.grid(True)
    plt.subplot(3, 3, i + 1)
    peaks_idx, troughs_idx = find_extramas(df, window_length=int(distance/2), width=int(distance/10), distance=15)
    plt.title(f"window_length {int(distance/2)} - width {int(distance/10)}")
    plt.xticks(np.linspace(0, 10, 5))
    plt.yticks(np.linspace(-2, 2, 5))
    price, = plt.plot(df.index, df.close, c='grey', lw=2, alpha=0.5, zorder=5)
    price_smooth, = plt.plot(df.index, df.close_smooth, c='b', lw=2, zorder=5)
    peaks, = plt.plot(df.index[peaks_idx], df.close_smooth.iloc[peaks_idx], c="r", linestyle='None', markersize = 10.0, marker = "o", zorder=10)
    troughs, = plt.plot(df.index[troughs_idx], df.close_smooth.iloc[troughs_idx], c="g", linestyle='None', markersize = 10.0, marker = "o", zorder=10)

plt.tight_layout()
plt.show()





