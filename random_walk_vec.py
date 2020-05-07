"""
random_walk_vec.py

Ben 
Jose Guzman, jose.guzman<at>guzman-lab.com

Created:Thu May  7 09:47:31 CEST 2020

This file contains functions to perform draw values from 
random distributions and analyze its effects.
"""

import numpy as np
#import pandas as pd (jguzman) if don't use, we don't import


# (jguzman) drift, sd_rw are not used, if param>3 use a dictionary
def random_walk_vec(n_reps, n_samples, drift, sd_rw, threshold):
    """ vectorized random walk model with np

    Parameters
    ----------
    n_reps : int
        number of normal distributions generated

    n_samples : int
        the number of samples drawn from a normal distribution

    drift : float 

    sd_rw : float

    threshold : float
    
    Returns
    -------
    A 1D NumPy array with the latency (in number of samples) of 
    the crossing values. Note that the size of the array is n_reps.
    """

    # TODO: (jguzman) do not accumulate object methods, it's confusing
    evidence = np.random.normal(0,
                                0.3,
                                [n_reps, n_samples]).cumsum(axis=1)

    # TODO: (jguzman) two objects with the same name is confusing
    evidence = np.concatenate((np.zeros((n_reps, 1)), evidence),
                              axis=1)

    # TODO: (jguzman) lambdas better be defined alone
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
    # TODO: (jguzman) where is the return value of this function????


if __name__ == '__main__':
    # just for testing run the script with :
    # python random_walk_vect.py

