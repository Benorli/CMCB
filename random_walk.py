import numpy as np
import pandas as pd

# TODO, create as df?
# TODO, np.vectorize > list comp > map > for, can this be implemented in a SIMPLE replicable way for diff models?


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
    rw_df = pd.DataFrame(data=list_trials,
                         columns=['evidence_accumulation',
                                  'trial_latencies',
                                  'trial_responses'])
    rw_df.index.name = 'trial'
    return rw_df


def simulate_trial(trial_evidence, n_samples, drift, sd_rw, threshold):
    """simulate a single trial of the random walk model"""
    trial_evidence[:] = np.cumsum(np.concatenate([np.zeros(1),
                                                  np.random.normal(drift,
                                                                   sd_rw,
                                                                   n_samples)]))
    trial_latency = np.where(abs(trial_evidence) > threshold)  # TODO: Clear values following the threshold
    assert trial_latency[0].size, "No decision made, sd_rw too low or threshold too high"
    trial_response = np.sign(trial_evidence[trial_latency])
    return trial_evidence, trial_latency[0][1], trial_response[0]
