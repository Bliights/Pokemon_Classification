from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from background import remove_background
from config import BackgroundMethod
from preprocessing import preprocessing
from utils import load_image

torch.manual_seed(42)


class PokemonDataset(Dataset):
    def __init__(
        self,
        dataset: pd.Series,
        labels: np.ndarray,
        background_method: BackgroundMethod,
        transform: Callable | None = None,
    ) -> None:
        """
        Parameters
        ----------
        dataset : pd.Series
            Series containing image file paths
        labels : np.ndarray
            Array containing encoded class labels corresponding to each image
        background_method : BackgroundMethod
            Background removal strategy to apply during preprocessing
        transform : Callable | None, optional
            Torchvision transform applied after preprocessing and background removal
        """
        self.dataset = dataset.reset_index(drop=True)
        self.labels = labels
        self.background_method = background_method
        self.transform = transform

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset

        Returns
        -------
        int
            Total number of images
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a single sample from the dataset

        Parameters
        ----------
        idx : int
            Index of the sample

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The image tensor and the corresponding label tensor
        """
        im_path = Path(self.dataset.iloc[idx])
        img = preprocessing(load_image(im_path))
        img = remove_background(img, im_path, method=self.background_method)

        img = transforms.functional.to_pil_image(img.astype(np.uint8))

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return img, label
