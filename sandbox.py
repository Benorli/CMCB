from Random_walk.random_walk import random_walk as rw

rw_df = rw(nreps=20000,
           nsamples=2000,
           drift=0,
           sd_rw=0.3,
           threshold=3)
print(rw_df)
