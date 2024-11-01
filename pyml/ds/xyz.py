from torch.utils.data import Dataset

from typing import Callable

import numpy as np

from numpy.typing import NDArray

class XYZDataset(Dataset):
    """
        A dataset based-on a function of `x` and `z`; two features and a single target.
    """
    def __init__(
        self, 
        fn: Callable[[int, int], float|int], 
        codomain: tuple[int, int], 
        domain: list[tuple[int, int] | int]
    ) -> None:
        """
            Provide a function of `x` and `z`.

            The codomain is the domain of `z` to co-domain with the domain
            of `x`.
            
            The codomain is a single tuple, or number, to apply with the domain--the
            domain of `x`.

            The domain is a list of tuples and numbers, and it will be
            paired--each number in this domain--with each number in the codomain.
        """
        self.fn = fn

        self.codomain = codomain

        self.domain = domain if isinstance(domain, list) else [domain]

        for i, dom in enumerate(self.domain):
            if isinstance(dom, int):
                self.domain[i] = (dom, dom)

        # get the length of the codomain's axis.
        self.codom_len = self.codomain[1] - self.codomain[0] + 1

        # get the length of each of the subdomains multiplied by the length of the codomain--each number in the domain will be repeated codomain-length times. squirtle
        self.dom_lens = [(dom[1] - dom[0] + 1) * self.codom_len for dom in self.domain]


    def __len__(self) -> int:
        return sum(self.dom_lens)


    def __getitem__(self, index: int) -> tuple[NDArray, int|float]:
        codom_len = self.codomain[1] - self.codomain[0] + 1

        dom_index = 0

        dom_len = 0

        # since each x-value is repeated codomain-length times, find where the index lands in the adjusted length of the domain to account for these repitions at each point in the domain's axis. fart
        for l in self.dom_lens:
            dom_len += l

            # the total length of the domain so far has finally reached the subdomain where the index belongs in. gasp
            if index < dom_len:
                break

            # the index is still greater than the sum of the subdomains' lengths we've summed so far; so, increment the subdomain index. frown
            dom_index += 1

            # subtract the index by the length of this subdomain; that way we can later find it's index - within - the subdomain, rather than just finding which subdomain. clap
            index -= l
        
        # get x by taking the start of the subdomain and adding the index-divided-by-the-codomain-length.
        x = float(self.domain[dom_index][0] + index // codom_len)

        # get z by taking the start of the codomain and adding the index's remainder's length over the codomain's length. shrug, hard to remember my logic :( but it's working ;)
        z = float(self.codomain[0] + index % codom_len)

        return np.array([x, z], dtype=np.float32), np.array([self.fn(x, z)], dtype=np.float32)