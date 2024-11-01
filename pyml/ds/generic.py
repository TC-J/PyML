from typing import Any, Callable

import numpy as np

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split


class GenericDataset(Dataset):
    """
        Create a generic dataset from data, or another dataset, that works with
        PyML functions and classes.

        The data is a `torch.Tensor`, an `numpy.ndarray`, or a `torch.utils.data.Dataset`.

        Specify `d_feature` and/or `d_target`, if the targets are more than one.

        This class will expect the targets to be the last columns of each row.

        Specify `transforms` as a list of function-calls on both the targets and the features.

        Use `feat_transforms` and `targ_transforms` the same way for transformations
        applied only to features or targets; further, when the list contains tuples of
        transforms, the individual columns will have the transformations applied; eg,
        when there are 3-features in each sample and you give a list of three-tuples containing callables,
        the transforms in each position of the tuple are applied to the individual features by column-position.
    """
    def __init__(
        self,
        data: torch.Tensor|np.ndarray|Dataset,
        d_feature: int | None = None,
        d_target: int | None = 1,
        transforms: list[Callable] | tuple[Callable] | Callable | None = None,
        feat_transforms: list[Callable] | list[tuple|None] | tuple[Callable] | Callable | None = None,
        targ_transforms: list[Callable] | list[tuple|None] | tuple[Callable] | Callable | None = None
    ):
        if not d_feature:
            d_feature = data[0].shape[-1] - d_target
        else:
            if data[0].shape[-1] != d_feature + d_target:
                d_target = data[0].shape[-1] - d_feature

        self._F = []

        self._T = []

        self._transforms: list[Callable] = transforms if isinstance(transforms, list) else [transforms]

        self._feat_transforms: list[Callable] | list[tuple[Callable, ...]] = feat_transforms if isinstance(feat_transforms, list) else [feat_transforms]

        self._targ_transforms: list[Callable] | list[tuple[Callable, ...]] = targ_transforms if isinstance(targ_transforms, list) else [targ_transforms]

        # if the data is a pytorch Dataset, get its contents into a list and then type-cast it to a numpy array.
        if isinstance(data, Dataset):
            for i in range(len(data)):
                f, t = data[i]

                self._F.append(f)

                self._T.append(t)

            self._F: np.ndarray[Any, Any] = np.asarray(self._F)

            self._T: np.ndarray[Any, Any] = np.asarray(self._T)
        # otherwise, its a Tensor or ndarray; so, slice the features and targets.
        else:
            self._F: torch.Tensor | np.ndarray = data[:, :d_feature]

            self._T: torch.Tensor | np.ndarray[Any, Any] = data[:, -d_target]
    
    
    def __len__(self) -> int:
        # return the number of rows in either the features or targets.
        return self._F.shape[0]

    
    def __getitem__(
        self,
        index: int
    ) -> tuple[torch.Tensor | np.ndarray | list | int | float, torch.Tensor | np.ndarray | list | int | float]:
        # get the features and targets for this index.
        f, t = self._F[index], self._T[index]

        # do not track the gradient.
        with torch.no_grad():
            for transform in self._transforms:
                f, t = transform(f), transform(t)

            # iterate and apply the feature-transforms.
            for feat_transform in self._feat_transforms:
                # if the transforms are in tuples, apply them to the feature's individual columns.
                if isinstance(feat_transform, tuple):
                    for i, _feat_transform in enumerate(feat_transform):
                        f[i] = _feat_transform(f[i])
                    # otherwise, apply the transform to the entire row.
                    else:
                        f = feat_transform(f)
            
            # iterate and apply the target-transforms.
            for targ_transform in self._targ_transforms:
                # if the transforms are in tuples, apply them to the target's individual columns.
                if isinstance(targ_transform, tuple):
                    for i, _targ_transform in enumerate(targ_transform):
                        t[i] = _targ_transform(t[i])
                    # otherwise, apply the transform to the entire row.
                    else:
                        t = targ_transform(t)
        return f, t


def generic_split_dataset(
    dataset: Dataset, 
    test_split: float = 0.2,
    transforms = None, 
    feat_transforms = None, 
    targ_transforms = None,
    **kwargs
) -> tuple[GenericDataset, GenericDataset]:
    """
        Create a training and testing `GenericDataset` from an existing PyTorch `Dataset`.

        Extra keyword-arguments are given to scikit-learn's `train_test_split`.
    """
    F, T = [], []

    for i in range(len(dataset)):
        f, t = dataset[i]

        F.append(f)

        T.append(t)
    
    F_train, T_train, F_test, T_test = train_test_split(F, T, test_split, **kwargs)

    return (
        GenericDataset(
            data=np.c_[np.asarray(F_train), np.asarray(T_train)],
            transforms=transforms,
            feat_transforms=feat_transforms,
            targ_transforms=targ_transforms
        ),
        GenericDataset(
            data=np.c_[np.asarray(F_test), np.asarray(T_test)],
            transforms=transforms,
            feat_transforms=feat_transforms,
            targ_transforms=targ_transforms
        )
    )