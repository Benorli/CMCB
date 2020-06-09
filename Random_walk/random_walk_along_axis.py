"""
random_walk_apply_along_axis.py

Ben

created: Tue Jun  2 15:52:59 2020

This file contains functions to perform a simple random
walk model of decision making. Uses apply_along_axis
method, slower than other methods.
"""

import numpy as np
import pandas as pd


def random_walk_apply_along_axis(nreps, nsamples, drift, sd_rw, threshold):
    """ Vectorized random walk model with np. Slower method,
        likely due to the usage of apply_along_axis.

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

    [*list_trials] = np.apply_along_axis(func1d=random_walk_trial,
                                         axis=1,
                                         arr=evidence,
                                         threshold=threshold)

    df_random_walk = pd.DataFrame(data=list_trials,
                                  columns=['evidence_accumulation',
                                           'trial_latencies',
                                           'trial_responses'])
    return df_random_walk


def random_walk_trial(evidence_row, threshold):
    """ Single trial for a random walk model of decision making.

    Parameters
    ----------
    evidence_row : numpy.ndarray
        A single row numpy array, containing a cumulative sum
        of random increments.

    threshold : float
        The value of evidence at which a decision is made.

    Returns
    -------
    evidence_row : numpy.ndarray
        A single row numpy array, containing a cumulative sum
        of random increments. Once a value reaches the threshold
        all following values are fixed.

    trial_latency : int
        The index which a decision is made (the evidence
        crosses a threshold). -1 if no decision is made.

    trial_response : int
        + 1 for decision made for upper threshold, -1
        when decision made for negative threshold, 0 for
        no decision made.
    """
    try:
        trial_latency = np.where(np.abs(evidence_row) >= threshold)[0][0]
    except IndexError:
        trial_latency = -1
    if trial_latency == -1:
        trial_response = 0
    else:
        trial_response = np.sign(evidence_row[trial_latency])
        evidence_row[trial_latency:] = trial_response * threshold
    return evidence_row, trial_latency, trial_response


if __name__ == '__main__':
    import timeit as tt
    import cProfile as cP

    print(tt.repeat("""df_rw = random_walk_apply_along_axis(nreps=2000,
                                                            nsamples=2000,
                                                            drift=0,
                                                            sd_rw=0.3,
                                                            threshold=3)""",
                    setup='from __main__ import random_walk_apply_along_axis',
                    repeat=5,
                    number=1))

    cP.run("""df_rw = random_walk_apply_along_axis(nreps=2000,
                                                   nsamples=2000,
                                                   drift=0,
                                                   sd_rw=0.3,
                                                   threshold=3)""")
