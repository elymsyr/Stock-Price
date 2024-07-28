import numpy as np
import matplotlib.pyplot as plt

# Sample time series data
np.random.seed(0)
time = np.arange(0, 10, 0.5)
values = np.sin(time) + np.random.normal(0, 0.5, len(time))

# Calculate the least squares error line
A = np.vstack([time, np.ones(len(time))]).T
m, c = np.linalg.lstsq(A, values, rcond=None)[0]

# Plot the time series data
plt.plot(time, values, label='Original data', markersize=10)
plt.plot(time, m*time + c, 'r', label='Least Squares Line')
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Time Series Data with Least Squares Line')
plt.legend()
plt.show()
