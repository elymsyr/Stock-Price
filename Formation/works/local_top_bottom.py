import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

def find_local_extrema(time_series, local_order):
    # Convert time series to numpy array for processing
    data = np.array(time_series)
    
    # Find local maxima and minima
    local_max = argrelextrema(data, np.greater)[0]
    local_min = argrelextrema(data, np.less)[0]
    
    # Limit the number of local maxima and minima to the local_order
    local_max = local_max[:local_order]
    local_min = local_min[:local_order]

    return local_max, local_min

def plot_time_series_with_extrema(time_series, local_max, local_min):
    plt.figure(figsize=(10, 6))
    plt.plot(time_series, label='Time Series Data')
    plt.scatter(local_max, time_series[local_max], color='red', label='Local Maxima', zorder=5)
    plt.scatter(local_min, time_series[local_min], color='blue', label='Local Minima', zorder=5)
    plt.title('Time Series with Local Extrema')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
np.random.seed(0)
time = np.arange(0, 10, 0.5)
time_series = np.sin(time) + np.random.normal(0, 0.5, len(time))
local_order = 100

local_max, local_min = find_local_extrema(time_series, local_order)
plot_time_series_with_extrema(time_series, local_max, local_min)
