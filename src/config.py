from enum import Enum
from pathlib import Path
from typing import Protocol, Self

import numpy as np

DATA_PATH = Path(__file__).resolve().parents[1] / "data"
DATASET_PATH = DATA_PATH / "fanart-dataset"
DATA_AUGMENTATION_PATH = DATA_PATH / "augmented"
BACKGROUND_PATH = DATA_PATH / "grabcut"


class ExtractMethod(Enum):
    SIFT = "sift"
    ORB = "orb"


class BackgroundMethod(Enum):
    NONE = None
    GRABCUT = "grabcut"


class AugmentationMethod(Enum):
    NONE = None
    TRUE = "flip/rotation"


class FitPredictModel(Protocol):
    def fit(self, x: np.ndarray, y: np.ndarray) -> Self: ...
    def predict(self, x: np.ndarray) -> np.ndarray: ...
