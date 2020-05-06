import numpy as np
import pandas as pd


def random_walk_vec(n_reps, n_samples, drift, sd_rw, threshold):
    """ vectorized random walk model with np"""

    evidence = np.random.normal(0,
                                0.3,
                                [n_reps, n_samples]).cumsum(axis=1)
    evidence = np.concatenate((np.zeros((n_reps, 1)), evidence),
                              axis=1)
    trial_latency = np.apply_along_axis(lambda x: np.where(x > threshold)[0][0], 1, np.abs(evidence))  # best solution for my purpose

    # QUESTION: how can I do this when evidence sometimes doesn't contain value I look for with where?

    # trial_latency = np.apply_along_axis(lambda x: np.where(x > threshold)[0][0], 1, evidence) ERROR
    # trial_latency = np.apply_along_axis(lambda x: np.where(x > threshold)[0], 1, evidence) ERROR

    # trial_latency = np.apply_along_axis(np.argmax, 1, evidence > threshold) gives this error:
    # https://stackoverflow.com/questions/45765476/why-does-numpy-argmax-for-a-list-of-all-false-bools-yield-zero/45765513
    # concatenated zeros prevent this at least

    trial_response = np.sign(evidence[:, trial_latency])

    # I will then put these into a dataframe and return them...

    # TODO: numpy.put to replace values above threshold
