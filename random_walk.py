import numpy as np

# TODO consider adding an input for data frame (may not be relevant, could input array into df)


def random_walk(n_reps, n_samples, drift, sd_rw, threshold):
    """random walk model"""

    # star unpacks generator (map output)
    [*a] = map(lambda trial: foo(trial,
                                 n_samples,
                                 drift,
                                 sd_rw,
                                 threshold),
               np.zeros((n_reps,
                         n_samples + 1)))
    return a


def foo(trial, n_samples, drift, sd_rw, threshold):
    trial[:] = np.cumsum(np.concatenate([np.zeros(1),
                                         np.random.normal(drift,
                                                          sd_rw,
                                                          n_samples)]))
    trial_latency = np.where(abs(trial) > threshold)
    trial_latency = trial_latency[0][1]
    trial_response = np.sign(trial[trial_latency])
    return trial, trial_latency, trial_response

