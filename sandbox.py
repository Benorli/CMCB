from Random_walk.random_walk import random_walk as rw
import seaborn as sns
import matplotlib.pyplot as plt

df_dv, df_trial_data = rw(nreps=2000,
                          nsamples=1000,
                          drift=0,
                          sd_rw=0.3,
                          threshold=3)

sns.set(style="darkgrid")
sns.lineplot(data=df_dv.iloc[:, 0:4], dashes=False)
plt.show()
