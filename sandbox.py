# from Random_walk.random_walk import random_walk as rw
import numpy as np


def every_more_than_each(every, each):
    every_min = np.min(every)
    more_than_all_min = each >= every_min
    if any(more_than_all_min):
        return False
    else:
        return True
