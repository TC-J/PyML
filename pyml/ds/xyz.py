from torch.utils.data import Dataset

from typing import Callable

import numpy as np

from numpy.typing import NDArray

class XYZDataset(Dataset):
    def __init__(
        self, 
        fn: Callable[[int, int], float|int], 
        codomain: tuple[int, int], 
        domain: list[tuple[int, int] | int]
    ) -> None:
        self.fn = fn

        self.codomain = codomain

        self.domain = domain if isinstance(domain, list) else [domain]

        for i, dom in enumerate(self.domain):
            if isinstance(dom, int):
                self.domain[i] = (dom, dom)

        self.codom_len = self.codomain[1] - self.codomain[0] + 1

        self.dom_lens = [(dom[1] - dom[0] + 1) * self.codom_len for dom in self.domain]


    def __len__(self) -> int:
        return sum(self.dom_lens)


    def __getitem__(self, index: int) -> tuple[NDArray, int|float]:
        codom_len = self.codomain[1] - self.codomain[0] + 1

        dom_index = 0

        dom_len = 0

        for l in self.dom_lens:
            dom_len += l

            if index < dom_len:
                break

            dom_index += 1

            index -= l
        
        x = float(self.domain[dom_index][0] + index // codom_len)

        y = float(self.codomain[0] + index % codom_len)

        return np.array([x, y], dtype=np.float32), np.array([self.fn(x, y)], dtype=np.float32)