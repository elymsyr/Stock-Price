from scipy.spatial.distance import cosine
import numpy as np

def get_score_cosine(formation: list, expected_list: np.ndarray) -> float:
    # Convert ndarray to list
    list2 = expected_list.flatten().tolist()

    # Ensure both lists are of the same length
    min_len = min(len(formation), len(list2))
    formation = formation[:min_len]
    list2 = list2[:min_len]

    # Calculate cosine similarity
    score = 1 - cosine(formation, list2)
    return float(score)