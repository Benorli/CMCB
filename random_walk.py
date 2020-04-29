import numpy as np

# TODO consider adding an input for data frame (may not be relevant, could input array into df)

# TODO consider adding an input for dataframe


def random_walk(n_reps, n_samples, drift, sdrw):
    """random walk model"""

    latencies = np.repeat(0, n_reps)  # TODO: remove predefine, predefine not necessary with map
    responses = np.repeat(0, n_reps)
    evidence = np.zeros((n_reps, n_samples + 1))
    # start unpacks generator (map output)
    [*a] = map(lambda trial_evidence: foo(trial_evidence, n_samples, drift, sdrw),
               evidence, responses, latencies)
    return a


def foo(trial_evidence, trial_response, trial_latency, n_samples, drift, sdrw):
    trial_evidence = np.cumsum(np.concatenate([np.zeros(1), np.random.normal(drift, sdrw, n_samples)]))
    trial_latency = np.where(abs(trial_evidence) > criterion)
    trial_latency = list(zip(trial_latency[0], trial_latency[1]))[0]
    trial_response = np.sign(trial_evidence[trial_latency])
    return trial_evidence, trial_latency, trial_response

