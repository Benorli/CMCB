import numpy as np

# TODO consider adding an input for data frame (may not be relevant, could input array into df)


def random_walk(n_reps, n_samples, drift, sd_rw, threshold):
    """random walk model"""

    # star unpacks generator (map output)
    [*list_trials] = map(lambda trial_evidence: simulate_trial(trial_evidence,
                                                               n_samples,
                                                               drift,
                                                               sd_rw,
                                                               threshold),
                         np.zeros((n_reps,
                                   n_samples + 1)))
    evidence = list(map(lambda x: x[0], list_trials))
    trial_latencies = list(map(lambda x: x[1], list_trials))
    trial_responses = list(map(lambda x: x[2], list_trials))
    return evidence, trial_latencies, trial_responses


def simulate_trial(trial_evidence, n_samples, drift, sd_rw, threshold):
    """simulate a single trial of the random walk model"""
    trial_evidence[:] = np.cumsum(np.concatenate([np.zeros(1),
                                                  np.random.normal(drift,
                                                                   sd_rw,
                                                                   n_samples)]))
    trial_latency = np.where(abs(trial_evidence) > threshold)
    trial_response = np.sign(trial_evidence[trial_latency])
    return trial_evidence, trial_latency[0][1], trial_response
