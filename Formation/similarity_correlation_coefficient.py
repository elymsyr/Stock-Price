from scipy.stats import pearsonr
import numpy as np
def get_score_pearson(formation: list, expected_list: np.ndarray) -> float:
    # Convert ndarray to list
    list2 = expected_list.flatten().tolist()

    # Ensure both lists are of the same length
    min_len = min(len(formation), len(list2))
    formation = formation[:min_len]
    list2 = list2[:min_len]

    # Calculate Pearson correlation coefficient
    corr, _ = pearsonr(formation, list2)
    print(f"Pearson Correlation Coefficient: {corr}")

    return float(corr) # type: ignore
