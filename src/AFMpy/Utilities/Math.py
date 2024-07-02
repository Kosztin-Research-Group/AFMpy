import numpy as np
from typing import Tuple

__all__ = ['norm']

def norm(x: np.ndarray,
         range: Tuple[float, float] = (0,1)):
    """
    Linearly rescales values of an array between specified minimum and maximum values

    Args:
        x (np.ndarray):
            Array of values to be rescaled.
        range (Tuple[float, float]):
            Tuple of minimum and maximum values to rescale the array between. Default is (0,1)
    Returns:
        np.ndarray:
            Rescaled Array
    """
    new_min, new_max = range
    return  (x - np.min(x)) / (np.max(x) - np.min(x)) * (new_max - new_min) + new_min
