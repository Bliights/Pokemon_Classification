import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from background import remove_background
from bovw import BoVW
from config import AugmentationMethod, BackgroundMethod, ExtractMethod, FitPredictModel
from data_augmentation import data_augmentation
from features import extract_descriptors
from logging_config import disable_logging, setup_logging
from preprocessing import preprocessing
from utils import load_image

logger = logging.getLogger(__name__)
setup_logging(logging.INFO)


def is_notebook() -> bool:
    """
    Detect if the code is running inside a Jupyter notebook.
    """
    return "ipykernel" in sys.modules


if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def _get_descriptors(
    dataset: pd.Series,
    extract_method: ExtractMethod,
    background_method: BackgroundMethod,
    max_features: int,
    display: str,
) -> list[np.ndarray]:
    """
    Extract local descriptors (SIFT/ORB) for each image in the dataset

    Parameters
    ----------
    dataset : pd.Series
        Series containing image file paths
    extract_method : ExtractMethod
        Descriptor extraction method
    background_method : BackgroundMethod
        Background segmentation method
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
            im_path = Path(p)
            img = preprocessing(load_image(im_path))
            img = remove_background(img, im_path, method=background_method)
            desc = extract_descriptors(img, method=extract_method, max_features=max_features)
            desc_list.append(desc)
            pbar.update(1)
    return desc_list


def split_dataset(
    dataset: pd.Series,
    label: pd.Series,
    augmentation_method: AugmentationMethod,
    extract_method: ExtractMethod,
    background_method: BackgroundMethod,
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
    augmentation_method : AugmentationMethod
        Data augmentation method
    extract_method : ExtractMethod
        Descriptor extraction method
    background_method : BackgroundMethod
        Background segmentation method
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

    x_train_paths, y_train = data_augmentation(x_train_paths, y_train, augmentation_method)

    logger.info(
        f"Dataset augmentation done ! (train={len(x_train_paths)}, test={len(x_test_paths)})",
    )

    desc_train = _get_descriptors(
        x_train_paths,
        extract_method,
        background_method,
        max_features,
        "train",
    )
    desc_test = _get_descriptors(
        x_test_paths,
        extract_method,
        background_method,
        max_features,
        "test",
    )

    logger.info("Starting BoVW training...")

    bovw = BoVW(n_words)
    bovw.fit(desc_train)

    logger.info("BoVW training finished ! ")
    logger.info("Starting BoVW encoding...")

    x_train = bovw.transform(desc_train)
    x_test = bovw.transform(desc_test)

    logger.info("BoVW encoding done ! ")

    return x_train, x_test, y_train, y_test


@disable_logging(logging.INFO)
def evaluate_all_methods(
    dataset: pd.Series,
    labels: pd.Series,
    label_encoder: LabelEncoder,
    models: list[FitPredictModel, dict | None],
) -> pd.DataFrame:
    """
    Evaluate all combinations of models, augmentation strategies, feature extraction methods
    and background removal methods

    Parameters
    ----------
    dataset : pd.Series
        Series containing image file paths
    labels : pd.Series
        Series containing the corresponding Pokemon labels
    label_encoder : LabelEncoder
        Fitted LabelEncoder used to retrieve class names for the
        classification report
    models : list[FitPredictModel, dict  |  None]
        List of tuples containing the model class to instantiate and the optional
        dictionary of initialization parameters

    Returns
    -------
    pd.DataFrame
        A dataframe containing the result for each configuration
    """
    results = []
    total = len(models) * len(AugmentationMethod) * len(ExtractMethod) * len(BackgroundMethod)

    with tqdm(
        total=total,
        desc="Evaluation",
        bar_format="{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} {postfix}",
        colour="blue",
    ) as pbar:
        for model_class, model_params in models:
            for augmentation in AugmentationMethod:
                for extract in ExtractMethod:
                    for background in BackgroundMethod:
                        pbar.set_postfix_str(
                            f"model={model_class.__name__} | "
                            f"augmentation={augmentation.name} | "
                            f"extract={extract.name} | "
                            f"background={background.name}",
                        )

                        x_train, x_test, y_train, y_test = split_dataset(
                            dataset,
                            labels,
                            augmentation,
                            extract,
                            background,
                        )

                        model: FitPredictModel = model_class(**model_params)

                        start = time.time()
                        model.fit(x_train, y_train)
                        train_time = time.time() - start

                        y_pred = model.predict(x_test)

                        results.append(
                            {
                                "model": model_class.__name__,
                                "augmentation_method": augmentation.name,
                                "extract_method": extract.name,
                                "background_method": background.name,
                                "accuracy": accuracy_score(y_test, y_pred),
                                "precision_macro": precision_score(
                                    y_test,
                                    y_pred,
                                    average="macro",
                                    zero_division=0,
                                ),
                                "recall_macro": recall_score(
                                    y_test,
                                    y_pred,
                                    average="macro",
                                    zero_division=0,
                                ),
                                "f1_macro": f1_score(
                                    y_test,
                                    y_pred,
                                    average="macro",
                                    zero_division=0,
                                ),
                                "classification_report": classification_report(
                                    y_test,
                                    y_pred,
                                    target_names=label_encoder.classes_,
                                    zero_division=0,
                                ),
                                "train_time_sec": train_time,
                            },
                        )
                        pbar.update(1)

    return pd.DataFrame(results)
