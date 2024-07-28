import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def generate_time_series(data_points, start_time='2024-01-01', frequency = 'ME'):
    """
    Generate a time series DataFrame from a list of data points, assuming each point represents a monthly interval.
    
    :param data_points: List of data points.
    :param start_time: Starting timestamp for the time series.
    :return: Pandas DataFrame with time series data.
    """
    time_index = pd.date_range(start=start_time, periods=len(data_points), freq=frequency)
    time_series = pd.DataFrame({'Timestamp': time_index, 'Value': data_points})
    return time_series

def interpolate_time_series(time_series, daily_volatility=0.025):
    """
    Interpolate the time series by generating daily data points using a random walk.
    
    :param time_series: Original time series DataFrame.
    :param daily_volatility: Standard deviation for the random walk (daily volatility).
    :return: Interpolated time series DataFrame with daily data points.
    """
    interpolated_data = []
    
    for i in range(len(time_series) - 1):
        start = time_series.iloc[i]
        end = time_series.iloc[i + 1]
        days_diff = (end['Timestamp'] - start['Timestamp']).days

        interpolated_data.append(start)

        # Calculate daily returns using a drift and volatility
        drift = (end['Value'] / start['Value']) ** (1 / days_diff) - 1
        daily_returns = np.random.normal(loc=drift, scale=daily_volatility, size=days_diff)
        daily_values = start['Value'] * np.exp(np.cumsum(daily_returns))

        for j in range(1, days_diff):
            t = start['Timestamp'] + pd.Timedelta(days=j)
            interpolated_data.append({'Timestamp': t, 'Value': daily_values[j - 1]})

    interpolated_data.append(time_series.iloc[-1])
    
    data = []
    # Add the last original point
    for item in interpolated_data:
        if isinstance(item, pd.Series):
            # Extracting values from the Series
            data.append({'Timestamp': item['Timestamp'], 'Value': item['Value']})
        elif isinstance(item, dict):
            # Assuming item is already in the correct format
            data.append(item)
    
    interpolated_df = pd.DataFrame(data)
    
    return interpolated_df

def find_local_extrema(data, time, distance):
    # Find indices of local maxima
    maxima_indices, _ = find_peaks(data, distance=distance)
    # Find indices of local minima by inverting the data
    minima_indices, _ = find_peaks([-x for x in data], distance=distance)
    
    maxima = [(time[i], data[i]) for i in maxima_indices]
    minima = [(time[i], data[i]) for i in minima_indices]
    
    return maxima, minima