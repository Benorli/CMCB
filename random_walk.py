import numpy as np

# set example params
n_reps = 10000
n_samples = 2000
drift = 0
sdrw = 0.3  # standard deviation random walk
criterion = 3

# TODO consider adding an input for dataframe


def random_walk(n_reps, n_samples, drift, sdrw):
    """random walk model"""

    latencies = np.repeat(0, n_reps)
    responses = np.repeat(0, n_reps)
    evidence = np.zeros((n_reps, n_samples + 1))
    # start unpacks generator (map output)
    [*a] = map(lambda evi: foo(evi, n_samples, drift, sdrw), enumerate(evidence))  # TODO remove unnecessary enumerate
    return a


def foo(evi, n_samples, drift, sdrw):
    idx, evidence = evi
    evidence = np.cumsum(np.concatenate([np.zeros(1), np.random.normal(drift, sdrw, n_samples)]))
    # TODO, p <− which ( abs (evidence[i ,])>criterion) [1] cross boundary
    return evidence


print(random_walk(n_reps, n_samples, drift, sdrw))
