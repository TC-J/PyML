from torch.utils.data import Dataset, DataLoader

from typing import Callable

import torch

import numpy as np

from numpy.typing import NDArray, DTypeLike

import pandas as pd


class XYDataset(Dataset):
    """
    """
    def __init__(
        self, 
        fn: Callable[[int|float], float|int],       # model's algebra to fn.
        domain: list[tuple[int, int] | int],   # the domain, or subdomains, over an assumed field of real or floats.
        dtype: DTypeLike = np.float32
    ) -> None:
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


def xy_gen_data(
    fn = lambda x: 2*x,
    domain = (-500, 500),
    dtype = np.float32
) -> pd.DataFrame:
    X = np.arange(
        start=domain[0], 
        stop=domain[1] + 1, 
        dtype=dtype
    )

    y = fn(X)

    return pd.DataFrame(data=np.c_[X, y], columns=["x", "y"])


def xy_gen_dl(
    fn = lambda x: 2*x,
    train_domain = [(-499, -200), (-99, 99), (200, 499)],
    test_domain = [(-199, -100), (100, 199)],
    xscale = None,
    yscale = None,
    shuffle: tuple[bool, bool] = (True, True),
    batch_size: tuple[int, int] = (256, 256),
    dtype = np.float32,
):
    return (
        DataLoader(
            dataset = XYDataset(
                fn = fn,
                domain = train_domain,
                dtype = dtype,
                xscale = xscale,
                yscale = yscale
            ),
            shuffle = shuffle[0],
            batch_size = batch_size[0]
        ),

        DataLoader(
            dataset = XYDataset(
                fn = fn,
                domain = test_domain,
                dtype = dtype,
                xscale = xscale,
                yscale = yscale
            ),
            shuffle = shuffle[1],
            batch_size = batch_size[1]
        )
    )