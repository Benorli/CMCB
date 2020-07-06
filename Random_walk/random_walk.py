"""
random_walk.py

Ben

created: Wed Jun  3 16:08:56 2020

This file contains functions to perform a simple random
walk model of decision making. List comprehension method was
final choice for jupyter notebook, for both speed and
readability.
"""

import numpy as np
import pandas as pd


def random_walk(nreps, nsamples, drift, sd_rw, threshold):
    """ Vectorized random walk model with np.

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
    df_dv : pandas.core.frame.DataFrame
        A DataFrame containing a column per trial, row per time point.
        Data represents the decision variable (internal evidence).
    df_trial_data : pandas.core.frame.DataFrame
        A DataFrame containing two columns trial_latency and
         trial_response. Note that the length of columns is n_reps.
    """
    # construct evidence accumulator for every trial

    start_zero_evidence = np.zeros((nreps, 1))
    rand_norm_incr = np.random.normal(loc=drift,
                                      scale=sd_rw,
                                      size=[nreps, nsamples])
    evidence_incr = np.concatenate((start_zero_evidence, rand_norm_incr),
                                   axis=1)
    acc_evidence = evidence_incr.cumsum(axis=1)

    dv, trial_latency, trial_response = zip(*[random_walk_trial(acc_evidence_row, threshold)
                                              for acc_evidence_row
                                              in acc_evidence])
    dv_array = np.asarray(dv).T
    column_names = ["trial_" + str(trial_n) for trial_n in np.arange(nreps)]

    df_dv = pd.DataFrame(data=dv_array,
                         columns=column_names)
    df_trial_data = pd.DataFrame(data={'trial_latency': trial_latency,
                                       'trial_response': trial_response})
    return df_dv, df_trial_data


def random_walk_trial(acc_evidence_row, threshold):
    """ Single trial for a random walk model of decision making.

    Parameters
    ----------
    acc_evidence_row : numpy.ndarray
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
        trial_latency = np.where(np.abs(acc_evidence_row) >= threshold)[0][0]  # idx 1st crossing threshold
    except IndexError:
        trial_latency = -1
    if trial_latency == -1:
        trial_response = 0
    else:
        trial_response = np.sign(acc_evidence_row[trial_latency])
        acc_evidence_row[trial_latency:] = trial_response * threshold  # fix after crossing threshold
    return acc_evidence_row, trial_latency, trial_response


if __name__ == '__main__':
    import timeit as tt
    import cProfile as cP

    print(tt.repeat("""df_rw = random_walk(nreps=20000,
                                           nsamples=2000,
                                           drift=0,
                                           sd_rw=0.3,
                                           threshold=3)""",
                    setup='from __main__ import random_walk',
                    repeat=5,
                    number=1))

    cP.run("""df_rw = random_walk(nreps=20000,
                                  nsamples=2000,
                                  drift=0,
                                  sd_rw=0.3,
                                  threshold=3)""")
