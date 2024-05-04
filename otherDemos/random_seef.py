import torch
import numpy as np
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
if __name__ == '__main__':
    # trainable params

    seed = 26
    setup_seed(seed)
    DATA_len = 10
    shuffled_indices = np.random.permutation(DATA_len)
    print(shuffled_indices)