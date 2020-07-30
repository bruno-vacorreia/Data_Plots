from tqdm import tqdm
import numpy as np
from scipy.signal import savgol_filter
from gnpy.core.utils import lin2db


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


def hz2thz(array_hz):
    """Convert Hz to THz"""
    return np.array(array_hz) / 1e12


def lin2dbm(value):
    return lin2db(value) + 30


def get_interval(x, y, y_min, y_max):
    x_new, y_new = [], []
    for i in range(0, len(y)-1):
        if y[i+1] >= y_min and y[i] <= y_max:
            y_new.append(y[i])
            x_new.append(x[i])
    x_new = np.array(x_new)
    y_new = np.array(y_new)

    return x_new, y_new


def smooth_curve(x, y, poly=3, div=2):
    x_new = x
    y_new = np.ravel(y)

    # Get odd value for window
    if (round(len(y_new)/div) % 2) == 0:
        window = round(len(y_new)/div) - 1
    else:
        window = round(len(y_new)/div)
    y_new = savgol_filter(x=y_new, window_length=window, polyorder=poly, mode='interp')

    return x_new, y_new
