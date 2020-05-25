"""
random_walk_vec.py

Ben 
Jose Guzman, jose.guzman<at>guzman-lab.com

Created:Thu May  7 09:47:31 CEST 2020

This file contains functions to perform draw values from 
random distributions and analyze its effects.
"""

import numpy as np
import pandas as pd
import seaborn as sns


# TODO: (jguzman) if param>3 use a dictionary
def random_walk_vec(nreps, nsamples, drift, sd_rw, threshold):
    """ vectorized random walk model with np

    Parameters
    ----------
    nreps : int
        number of normal distributions generated

    nsamples : int
        the number of samples drawn from a normal distribution

    drift : float
        The initial evidence

    sd_rw : float
        The standard deviation of a random normal distribution
        which defines the step in evidence made at each time
        point

    threshold : float
        The value of evidence at which a decision is made.
    
    Returns
    -------
    A DataFrame containing three columns, evidence, trial_latency,
    and trial_response. Note that the length of columns is n_reps.
    """

    evidence = np.concatenate((np.zeros((nreps, 1)),
                               np.random.normal(loc=drift,
                                                scale=sd_rw,
                                                size=[nreps, nsamples])),
                              axis=1)
    evidence[:] = evidence.cumsum(axis=1)

    trial_latency = np.apply_along_axis(func1d=where_first,
                                        axis=1,
                                        arr=np.abs(evidence) > threshold)
    trial_response = np.sign(evidence[:, trial_latency])

    df_random_walk = pd.DataFrame(data={'evidence': evidence,
                                        'trial_latency': trial_latency,
                                        'trial_response': trial_response})
    return df_random_walk

    # TODO: Finish and plot intention for clarity
    # TODO: numpy.put to replace values above threshold


def where_first(x):
    return np.where(x)[0][0]


def plot_random_walk(df_random_walk, trials):
    """ 
    Plots random walk model data

    Parameters
    ----------
    df_random_walk : pandas.core.frame.DataFrame
        Each row is a trial. Columns must include; evidence,
        trial_response, and trial_latency.

    trials : numpy.ndarray
        An array containing trials to be included.

    """
    # TODO: recreate var space and try plotting
    sns.set(style="darkgrid")
    sns.lineplot(data=df_random_walk.evidence[trials])


if __name__ == '__main__':
    # just for testing run the script with :
    # python random_walk_vect.py
    import matplotlib.pyplot as plt
    #
    # data = np.random.normal(loc = 0, scale = 0.3, size = (30, 5000))
    # mylat = get_latencies(n_reps = 30, n_samples = 1000, threshold=0.2)
    #
    # fig, ax = plt.subplots(1, 2, figsize = (12, 8))
    #
    # for trace in data:
    #     # get first the histograms
    #     values, base = np.histogram(trace, bins=40)
    #     # compute cumulative
    #     mycum = np.cumsum(values)
    #     ax[0].plot(base[:-1], mycum, c = 'gray', lw = 1)
    #
    # ax[0].set_ylabel('Cumulative distribution')
    # ax[1].hist(mylat, bins = np.arange(0,200, 5))
    # ax[1].set_ylabel('Number of ocurrences')
    # ax[1].set_xlabel('Latency (in sampling points)')
    #
    #
    # fig.show()
    # set example params
    n_reps = 1000
    n_samples = 2000
    drift = 0
    sd_rw = 0.3  # standard deviation random walk
    threshold = 3

    ax[0].set_ylabel('Cumulative distribution')
    ax[1].hist(mylat, bins = np.arange(0,200, 5))
    ax[1].set_ylabel('Number of ocurrences')
    ax[1].set_xlabel('Latency (in sampling points)')


    fig.show()



