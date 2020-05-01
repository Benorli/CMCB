import random_walk as rw

# set example params
n_reps = 10000
n_samples = 2000
drift = 0
sd_rw = 0.3  # standard deviation random walk
threshold = 3

evidence, trial_latencies, trial_responses = rw.random_walk(n_reps,
                                                            n_samples,
                                                            drift,
                                                            sd_rw,
                                                            threshold)
