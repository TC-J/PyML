import numpy as np
import torch
from torch.utils.data import Dataset

import sys

def dataset_min_max(dataset: Dataset):
    """
        Get the minimums and maximums for each feature and target in a dataset. 

        Returns, first, the array of feature-minimums, then, the feature-
        maximums, then the target-minimums, and then finally, the target-maximums--
        all ordered by column.
    """
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


def dataset_mean_std(dataset: Dataset):
    """
        Get the mean and standard deviation for each column of the feature(s) 
        and the target(s) and for the features and targets as a whole.

        Returns, first, the column-wise mean and then standard-deviation for the
        features and then the targets, respectively, followed by the total-features'
        mean and standard-deviation, and the same for the total-targets'.
    """
    F = []

    T = []

    # get all the data from the dataset in lists.
    for i in range(len(dataset)):
        f, t = dataset[i]

        # append the ndarray from the tensors or the immideate-values to the lists.
        F.append(f.detach().numpy() if isinstance(f, torch.Tensor) else f)

        T.append(t.detach().numpy() if isinstance(t, torch.Tensor) else t)
    
    # convert the lists to ndarrays. start
    F = np.asarray(F)

    T = np.asarray(T)

    return F.mean(axis=0), F.std(axis=0), T.mean(axis=0), T.std(axis=0), F.mean(), F.std(), T.mean(), T.std()