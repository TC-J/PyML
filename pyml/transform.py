import numpy as np

import torch

from abc import abstractmethod, ABC

class Normalization(ABC):
    @abstractmethod
    def normalize(
        self, 
        data: torch.Tensor | np.ndarray | int | float
    ) -> torch.Tensor | np.ndarray | int | float:
        pass

    @abstractmethod
    def denormalize(
        self,
        data: torch.Tensor | np.ndarray | int | float
    ) -> torch.Tensor | np.ndarray | int | float:
        pass


class Rescale(Normalization):
    def __init__(
        self,
        from_min: int | float,
        from_max: int | float,
        to_min: int | float,
        to_max: int | float
    ):
        self._from = (from_min, from_max)

        self._to = (to_min, to_max)
    
    
    def normalize(
        self,
        data: torch.Tensor | np.ndarray | int | float
    ) -> torch.Tensor | np.ndarray | int | float:
        with torch.no_grad():
            return ((self._to[1] - self._to[0]) * ((data - self._from[0]) / (self._from[1] - self._from[0]))) + self._to[0]
    
    
    def denormalize(
        self,
        data: torch.Tensor | np.ndarray | int | float
    ) -> torch.Tensor | np.ndarray | int | float:
        with torch.no_grad():
            return ((data - self._to[0]) / (self._to[1] - self._to[0]) * (self._from[1] - self._from[0])) + self._from[0]