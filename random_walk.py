import numpy as np
import pandas as pd


def random_walk(nreps, nsamples, drift, sd_rw, threshold):
    """random walk model"""
    # star unpacks generator (map output)
    [*list_trials] = map(lambda trial_evidence: simulate_trial(trial_evidence,
                                                               nsamples,
                                                               drift,
                                                               sd_rw,
                                                               threshold),
                         np.zeros((nreps,
                                   nsamples + 1)))
    rw_df = pd.DataFrame(data=list_trials,
                         columns=['evidence_accumulation',
                                  'trial_latencies',
                                  'trial_responses'])
    rw_df.index.name = 'trial'
    return rw_df


def simulate_trial(trial_evidence, nsamples, drift, sd_rw, threshold):
    """simulate a single trial of the random walk model"""
    trial_evidence[:] = np.cumsum(np.concatenate([np.zeros(1),
                                                  np.random.normal(drift,
                                                                   sd_rw,
                                                                   nsamples)]))
    trial_latency = np.where(abs(trial_evidence) > threshold)
    assert trial_latency[0].size, "No decision made, sd_rw too low or threshold too high"
    trial_response = np.sign(trial_evidence[trial_latency])
    return trial_evidence, trial_latency[0][1], trial_response[0]


if __name__ == '__main__':
    """Testing space"""

    import timeit as tt
    import cProfile as cP

    print(tt.repeat("""df_rw = random_walk(nreps=2000,
                                           nsamples=2000,
                                           drift=0,
                                           sd_rw=0.3,
                                           threshold=3)""",
                    setup='from __main__ import random_walk',
                    repeat=2,
                    number=1))

    cP.run("""df_rw = random_walk(nreps=2000,
                                  nsamples=2000,
                                  drift=0,
                                  sd_rw=0.3,
                                  threshold=3)""")
