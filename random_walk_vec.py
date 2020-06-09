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


def random_walk_vec(nreps, nsamples, drift, sd_rw, threshold):
    """ Vectorized random walk model with np. Explored focusing
        on vectorisation afforded by numpy functions. Results
        in the slowest method, likely due to the multiple usage
        of apply_along_axis.

    Parameters
    ----------
    nreps : int
        number of normal distributions generated.

    nsamples : int
        the number of samples drawn from a normal distribution.

    drift : float
        The initial evidence.

    sd_rw : float
        The standard deviation of a random normal distribution
        which defines the step in evidence made at each time
        point.

    threshold : float
        The value of evidence at which a decision is made.
    
    Returns
    -------
    out : pandas.core.frame.DataFrame
        A DataFrame containing three columns, evidence,
        trial_latency, and trial_response. Note that the
        length of columns is n_reps.
    """
    # construct evidence accumulator for every trial
    evidence = np.concatenate((np.zeros((nreps, 1)),
                               np.random.normal(loc=drift,
                                                scale=sd_rw,
                                                size=[nreps, nsamples])),
                              axis=1)
    evidence[:] = evidence.cumsum(axis=1)
    # index where the threshold was crossed
    trial_latency = np.apply_along_axis(func1d=where_first,
                                        axis=1,
                                        arr=abs(evidence) > threshold)
    trial_latency_top = np.apply_along_axis(func1d=where_first,
                                            axis=1,
                                            arr=evidence > threshold)
    trial_latency_bot = np.apply_along_axis(func1d=where_first,
                                            axis=1,
                                            arr=evidence < -threshold)
    # fix evidence to threshold once crossed
    idx = np.arange(nreps)
    evidence[idx > trial_latency_top.T] = threshold
    evidence[idx < trial_latency_bot.T] = -threshold
    # TODO: Check this section

    # combine top and bottom latency
    trial_latency = trial_latency_top
    trial_latency[trial_latency_bot < trial_latency_top] = trial_latency_bot[trial_latency_bot < trial_latency_top]
    trial_latency[trial_latency_top == -1] = trial_latency_bot[trial_latency_top == -1]
    # TODO: take min then deal with zeroes
    # TODO: when taking min, -1 is copied, need to deal with this
    # TODO: array is modified inplace! be careful! Should copy

    trial_response = np.sign(evidence[:, trial_latency])[:, 0]

    df_random_walk = pd.DataFrame(data={'evidence': list(evidence),
                                        'trial_latency': trial_latency,
                                        'trial_response': trial_response})
    return df_random_walk

    # TODO: Finish and plot intention for clarity
    # TODO: Replace values above threshold with ((start  <= r)) where r is np.arange and start is trial_latency
    #       may have to sep positive and negative values. where True give +- 3 then asign with mask


def where_first(x):
    try:
        first_idx = np.where(x)[0][0]
    except IndexError:
        first_idx = -1
    return first_idx


if __name__ == '__main__':
    import timeit as tt
    import cProfile as cP

    print(tt.repeat("""df_rw = random_walk_vec(nreps=2000,
                                               nsamples=2000,
                                               drift=0,
                                               sd_rw=0.3,
                                               threshold=3)""",
                    setup='from __main__ import random_walk_vec',
                    repeat=2,
                    number=1))

    cP.run("""df_rw = random_walk_vec(nreps=2000,
                                      nsamples=2000,
                                      drift=0,
                                      sd_rw=0.3,
                                      threshold=3)""")
