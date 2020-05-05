import numpy as np
import pandas as pd


def random_walk_vec(n_reps, n_samples, drift, sd_rw, threshold):
    """ vectorzied random walk model with np"""

    evidence = np.random.normal(0,
                                0.3,
                                [n_reps, n_samples]).cumsum(axis=1)
    evidence = np.concatenate((np.zeros((n_reps, 1)), evidence),
                              axis=1)
    # TODO: Next np.which
    # TODO: Use where to apply function depending on value
    # TODO: numpy.put to replace values above threshold
    # TODO: which may require apply along axis (could include put here too)
