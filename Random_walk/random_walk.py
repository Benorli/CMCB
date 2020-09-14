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


def gen_drift(evidence, sd_rw, nreps, nsamples):
    """
    Generate random drift, of a decision variable dependent on evidence

     Parameters
    ----------
    evidence : float
        The evidence given by the stimuli. The drift of the decision
        variable is defined by increments drawn from a random normal
        distribution with this mean value.

    sd_rw : float
        The drift of the decision variable is defined by increments
        drawn from a random normal distribution with this standard
        deviation.

    nreps : int
        The number of trials.

    nsamples : int
        The length of each trial.

    Returns
    -------
    acc_drift : numpy.ndarray
        A 2d numpy array representing the drift of the decision variable.
        Each row is a trial and each column is a sample (time point).
        """
    start_zero_drift = np.zeros((nreps, 1))
    rand_norm_incr = np.random.normal(loc=evidence,
                                      scale=sd_rw,
                                      size=[nreps, nsamples])
    drift_incr = np.concatenate((start_zero_drift, rand_norm_incr),
                                axis=1)
    acc_drift = drift_incr.cumsum(axis=1)
    return acc_drift


def gen_drift_with_noise(evidence, sd_rw, nreps, nsamples, start_noise, evidence_noise):
    """
    Generate random drift, of a decision variable dependent on evidence.
    Includes noise to the starting evidence and the standard deviation of the
    drift


     Parameters
    ----------
    evidence : float
        The evidence given by the stimuli. The drift of the decision
        variable is defined by increments drawn from a random normal
        distribution with this mean value.

    sd_rw : float
        The drift of the decision variable is defined by increments
        drawn from a random normal distribution with this standard
        deviation.

    nreps : int
        The number of trials.

    nsamples : int
        The length of each trial.

    start_noise : int
        The standard deviation of the noise on the starting point of the
        decision variable.

    evidence_noise : int

    Returns
    -------
    acc_drift : numpy.ndarray
        A 2d numpy array representing the drift of the decision variable.
        Each row is a trial and each column is a sample (time point).
        """
    trial_start = np.random.normal(loc=0,
                                   scale=start_noise,
                                   size=[nreps, 1])
    drift_sd = np.random.normal(loc=sd_rw,
                                scale=evidence_noise,
                                size=[nreps, nsamples])
    rand_norm_incr = np.random.normal(loc=evidence,
                                      scale=drift_sd,
                                      size=[nreps, nsamples])
    drift_incr = np.concatenate((trial_start, rand_norm_incr),
                                axis=1)
    acc_drift = drift_incr.cumsum(axis=1)
    return acc_drift


def random_walk(nreps, threshold, acc_drift):
    """
    Random walk model of decision making.

    Parameters
    ----------
    nreps : int
        The number of trials.

    threshold : float
        The value of evidence at which a decision is made.

    acc_drift : numpy.ndarray
        A numpy array representing the drift of the decision variable.

    Returns
    -------
    df_dv : pandas.core.frame.DataFrame
        A DataFrame containing a column per trial, row per time point.
        Data represents the decision variable (internal evidence).

    df_trial_data : pandas.core.frame.DataFrame
        A DataFrame containing two columns trial_latency and
         trial_response. Note that the length of columns is n_reps.

    """
    # run the random walk function on every row of drift
    dv, trial_latency, trial_response = zip(*[random_walk_trial(acc_drift_row, threshold)
                                              for acc_drift_row
                                              in acc_drift])

    column_names = ["trial_" + str(trial_n + 1) for trial_n in np.arange(nreps)]
    dv_array = np.asarray(dv).T
    df_dv = pd.DataFrame(data=dv_array,
                         columns=column_names)
    df_trial_data = pd.DataFrame(data={'trial_latency': trial_latency,
                                       'trial_response': trial_response})
    return df_dv, df_trial_data


def random_walk_trial(acc_drift_row, threshold):
    """
    Single trial for a random walk model of decision making.

    Parameters
    ----------
    acc_drift_row : numpy.ndarray
        A single row numpy array, containing a cumulative sum of random
        increments.

    threshold : float
        The value of evidence at which a decision is made.

    Returns
    -------
    acc_drift_row : numpy.ndarray
        A single row numpy array, containing a cumulative sum of random
        increments. Once a value reaches the threshold the trial is over
        and following values are set to NaN.

    trial_latency : int
        The index which a decision is made (the evidence crosses a
        threshold). -1 if no decision is made.

    trial_response : int
        + 1 for decision made for upper threshold, -1 when decision made
        for negative threshold, 0 for no decision made.
    """
    try:
        # look for the first index when the threshold is crossed
        trial_latency = np.where(np.abs(acc_drift_row) >= threshold)[0][0]
    # if the index is never crossed
    except IndexError:
        trial_latency = -1  # no latency
    if trial_latency == -1:
        trial_response = 0  # no response
    else:
        trial_response = np.sign(acc_drift_row[trial_latency])
        # After the decision is made the trial ends
        acc_drift_row[trial_latency + 1:] = np.NaN
    return acc_drift_row, trial_latency, trial_response


if __name__ == '__main__':
    # import timeit as tt
    # import cProfile as cP
    print('test not set')
    #
    # print(tt.repeat("""df_rw = random_walk(nreps=20000,
    #                                        nsamples=2000,
    #                                        drift=0,
    #                                        sd_rw=0.3,
    #                                        threshold=3)""",
    #                 setup='from __main__ import random_walk',
    #                 repeat=5,
    #                 number=1))
    #
    # cP.run("""df_rw = random_walk(nreps=20000,
    #                               nsamples=2000,
    #                               drift=0,
    #                               sd_rw=0.3,
    #                               threshold=3)""")
