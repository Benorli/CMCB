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

def get_latencies(n_reps, n_samples, threshold):
    """ 
    Calculates the latencies (in number of samples) of crossing
    threshodls of normaly distributed random variables with mean
    zero and standard deviation 0.3.

    Parameters
    ----------
    n_reps : int
        number of normal distributions generated

    n_samples : int
        the number of samples drawn from a normal distribution

    threshold : float
        the value above which the latency will be calculated
    
    Returns
    -------
    A 1D NumPy array with the latency (in number of samples) of 
    the crossing values. Note that the size of the array is n_reps.
    """
    mysize = (n_reps, n_samples)
    mynorm = np.random.normal(loc = 0, scale = 0.3, size = mysize)

    rows, cols = np.where(mynorm>threshold) # returns rows, cols

    # we simply take the columns up to the number of repetitions (n_reps)
    # because these are the first instances where the repetition was
    # above the threshold
    mylatency = cols[:n_reps] # we take up to the number of reps

    return mylatency

if __name__ == '__main__':
    # just for testing run the script with :
    # python random_walk_vect.py
    import matplotlib.pyplot as plt

    data = np.random.normal(loc = 0, scale = 0.3, size = (30, 5000))
    mylat = get_latencies(n_reps = 30, n_samples = 1000, threshold=0.2)

    fig, ax = plt.subplots(1, 2, figsize = (12, 8))

    for trace in data:
        # get first the histograms
        values, base = np.histogram(trace, bins=40)
        # compute cumulative 
        mycum = np.cumsum(values)
        ax[0].plot(base[:-1], mycum, c = 'gray', lw = 1)

    ax[0].set_ylabel('Cumulative distribution')
    ax[1].hist(mylat, bins = np.arange(0,200, 5))
    ax[1].set_ylabel('Number of ocurrences')
    ax[1].set_xlabel('Latency (in sampling points)')


    fig.show()



