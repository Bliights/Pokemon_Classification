from enum import Enum
from typing import Protocol, Self

import numpy as np


class ExtractMethod(Enum):
    SIFT = "sift"
    ORB = "orb"


class BackgroundMethod(Enum):
    NONE = None
    GRABCUT = "grabcut"


class FitPredictModel(Protocol):
    def fit(self, x: np.ndarray, y: np.ndarray) -> Self: ...
    def predict(self, x: np.ndarray) -> np.ndarray: ...
