import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from bovw import BoVW
from config import ExtractMethod
from features import extract_descriptors
from logging_config import setup_logging
from preprocessing import preprocessing
from utils import load_image

logger = logging.getLogger(__name__)
setup_logging(logging.INFO)


def _get_descriptors(
    dataset: pd.Series,
    method: ExtractMethod,
    max_features: int,
    display: str,
) -> list[np.ndarray]:
    """
    Extract local descriptors (SIFT/ORB) for each image in the dataset

    Parameters
    ----------
    dataset : pd.Series
        Series containing image file paths
    method : ExtractMethod
        Descriptor extraction method
    max_features : int
        Maximum number of keypoints/descriptors to keep per image

    Returns
    -------
    list[np.ndarray]
        List of descriptor matrices
    """
    desc_list = []
    total = len(dataset)

    with tqdm(
        total=total,
        desc=f"Extracting descriptors for {display}",
        bar_format="{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} {postfix}",
        colour="green",
    ) as pbar:
        for p in dataset:
            img = preprocessing(load_image(Path(p)))
            desc = extract_descriptors(img, method=method, max_features=max_features)
            desc_list.append(desc)
            pbar.update(1)
    return desc_list


def split_dataset(
    dataset: pd.Series,
    label: pd.Series,
    method: ExtractMethod,
    test_size: int = 0.2,
    max_features: int = 800,
    n_words: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split a dataset into train/test, then build BoVW features for both splits

    Parameters
    ----------
    dataset : pd.Series
        Series of image paths
    label : pd.Series
        Series of labels
    method : ExtractMethod
        Descriptor method
    test_size : int, optional
        Fraction of samples used as test set
    max_features : int, optional
        Maximum number of local descriptors per image
    n_words : int, optional
        Size of the BoVW vocabulary

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        x_train, x_test, y_train, y_test
    """
    x_train_paths, x_test_paths, y_train, y_test = train_test_split(
        dataset,
        label,
        test_size=test_size,
        random_state=42,
        stratify=label,
    )
    logger.info(
        f"Dataset split done ! (train={len(x_train_paths)}, test={len(x_test_paths)})",
    )

    desc_train = _get_descriptors(x_train_paths, method, max_features, "train")
    desc_test = _get_descriptors(x_test_paths, method, max_features, "test")

    logger.info("Starting BoVW training...")

    bovw = BoVW(n_words)
    bovw.fit(desc_train)

    logger.info("BoVW training finished ! ")
    logger.info("Starting BoVW encoding...")

    x_train = bovw.transform(desc_train)
    x_test = bovw.transform(desc_test)

    logger.info("BoVW encoding done ! ")

    return x_train, x_test, y_train, y_test
