from torch.utils.data import Dataset, DataLoader

from typing import Callable

import torch

import numpy as np

from numpy.typing import NDArray, DTypeLike

import pandas as pd


class XYDataset(Dataset):
    """
        Create a dataset based on a function of `x`; a single feature and target.
    """
    def __init__(
        self, 
        fn: Callable[[int|float], float|int],       # model's algebra to fn.
        domain: list[tuple[int, int] | int],   # the domain, or subdomains, over an assumed field of real or floats.
        dtype: DTypeLike = np.float32
    ) -> None:
        """
            Pass a function to create the targets from the domain of `x` values.

            The domain is a tuple, or a list of tuples, representing the subdomains 
            included in the domain.
        """
        self.__dtype = dtype

        self.__fn = fn

        self.__domain = domain if isinstance(domain, list) else [domain]

        # check if incoming domain is split into subdomains of said domain.
        for _i, _subdom in enumerate(self.__domain):
            if isinstance(_subdom, int):
                self.__domain[_i] = (_subdom, _subdom)

        self.__dom_lens = [(dom[1] - dom[0] + 1) for dom in self.__domain]


    def __len__(self) -> int:
        return sum(self.__dom_lens)


    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        __dom_index = 0

        __dom_len = 0

        for _l in self.__dom_lens:
            __dom_len += _l

            if index < __dom_len:
                break

            __dom_index += 1

            index -= _l

        x = float(self.__domain[__dom_index][0] + index)

        y = self.__fn(x)
        
        return torch.Tensor([x]), torch.Tensor([y])