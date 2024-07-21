FORMATIONS = {
    'ascending_triangle': [0.9, 0.1, 0.9, 0.52, 0.1, 0.26, 0.1],
    'descending_triangle': [0.9, 0.1, 0.5, 0.1, 0.29, 0.1],
    'symmetrical_triangle': [0.9, 0.1, 0.7, 0.3, 0.4, 0.15, 0.2, 0.11, 0.1]
}

# crate a function that gets the data (monthly or yearly or something else)  and find the best formation that fit.

import numpy as np
from scipy.spatial.distance import euclidean

from fastdtw import fastdtw # type: ignore

x = np.array([[1.1], [2.2], [3.3], [4.4], [5.5]])
y = np.array([[2.2], [3.3], [4.4]])

distance, path = fastdtw(x, y, dist=euclidean)
print(distance, path)