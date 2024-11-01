import numpy as np

import torch

from abc import abstractmethod, ABC

class Transformation(ABC):
    """
        Abstract base class for classes that provide transformation. 
    """
    @abstractmethod
    def transform(
        self, 
        data: torch.Tensor | np.ndarray | int | float
    ) -> torch.Tensor | np.ndarray | int | float:
        pass

    @abstractmethod
    def inverse(
        self,
        data: torch.Tensor | np.ndarray | int | float
    ) -> torch.Tensor | np.ndarray | int | float:
        pass


class Normalize(Transformation):
    """
        Normalize data by scaling it to a specific minimum and maximum, using 
        the minimum and maximum of the input-data.
    """
    def __init__(
        self,
        from_min: int | float,
        from_max: int | float,
        to_min: int | float,
        to_max: int | float
    ):
        """
            Pass the min and max and the rescaled min and max for normalization.
        """
        self._from = (from_min, from_max)

        self._to = (to_min, to_max)
    
    
    def transform(
        self,
        data: torch.Tensor | np.ndarray | int | float
    ) -> torch.Tensor | np.ndarray | int | float:
        """
            Normalize the data based on the rescale.
        """
        with torch.no_grad():
            return ((self._to[1] - self._to[0]) * ((data - self._from[0]) / (self._from[1] - self._from[0]))) + self._to[0]
    
    
    def inverse(
        self,
        data: torch.Tensor | np.ndarray | int | float
    ) -> torch.Tensor | np.ndarray | int | float:
        """Reverse the normalization of the data based on this rescale."""
        with torch.no_grad():
            return ((data - self._to[0]) / (self._to[1] - self._to[0]) * (self._from[1] - self._from[0])) + self._from[0]
    

class Standardize(Transformation):
    def transform():
        pass

    def inverse():
        pass