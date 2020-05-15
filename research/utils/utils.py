from tqdm import tqdm
import numpy as np


def identify_pareto_max(scores: np.array):
    """Function to identify the index of the non-dominated solution for two objectives"""
    # Count number of items
    population_size = scores.shape[0]

    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)

    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Pareto front
    pareto_front = np.ones(population_size, dtype=bool)

    # Loop through each item. This will then be compared with all other items
    pop_tqdm = tqdm(iterable=range(population_size), desc='Computing pareto')

    for i in pop_tqdm:
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break

    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]


def hz2thz(array_hz: np.array):
    return array_hz / 1e12
