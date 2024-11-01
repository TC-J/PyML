import torch
from torch.utils.data import Dataset

import sys

def dataset_min_max(dataset: Dataset):
    _f, _t = dataset[0]

    f_mins, t_mins = [sys.float_info.max] * _f.shape[-1], [sys.float_info.max]  * _t.shape[-1]

    f_maxs, t_maxs = [sys.float_info.min] * _f.shape[-1], [sys.float_info.min] * _t.shape[-1]

    for index in range(len(dataset)):
        F, T = dataset[index]

        for i, f in enumerate(F):
            if f < f_mins[i]:
                f_mins[i] = f

            elif f > f_maxs[i]:
                f_maxs[i] = f
        
        for i, t in enumerate(T):
            if t < t_mins[i]:
                t_mins[i] = t

            elif t > t_maxs[i]:
                t_maxs[i] = t

    return f_mins, f_maxs, t_mins, t_maxs
