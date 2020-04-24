import numpy as np

# set example params
n_reps = 10000
n_samples = 2000
drift = 0
sdrw = 0.3
criterion = 3


def random_walk(n_reps, n_samples, drift, sdrw):
    """random walk model"""

    latencies = np.repeat(0, n_reps)
    responses = np.repeat(0, n_reps)
    evidence = np.zeros((n_reps, n_samples + 1))
    map(sum, evidence)  # INCOMPLETE, sum is not correct function continue from here
