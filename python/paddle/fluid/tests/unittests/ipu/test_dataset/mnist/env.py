import random
import numpy as np

def set_random_seed(seed):
    # for paddle >= 1.8.5
    random.seed(seed)
    np.random.seed(seed)

