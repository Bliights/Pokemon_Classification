import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torchvision import transforms

from background import remove_background
from bovw import BoVW
from config import AugmentationMethod, BackgroundMethod, ExtractMethod, FitPredictModel
from data_augmentation import data_augmentation
from dataset import PokemonDataset
from features import extract_descriptors
from logging_config import disable_logging, setup_logging
from models import BaseImageModel
from preprocessing import preprocessing
from utils import load_image

logger = logging.getLogger(__name__)
setup_logging(logging.INFO)

torch.manual_seed(42)

g = torch.Generator()
g.manual_seed(42)


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
        leave=False,
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
    test_size: float = 0.2,
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
    test_size : float, optional
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


def split_dataset_torch(
    dataset: pd.Series,
    labels: pd.Series,
    augmentation_method: AugmentationMethod,
    background_method: BackgroundMethod,
    test_size: float = 0.2,
    val_size: float = 0.1,
    input_size: int = 224,
) -> tuple[PokemonDataset, PokemonDataset, PokemonDataset]:
    """
    Split a dataset into train, validation and test sets and create the corresponding PyTorch Dataset

    Parameters
    ----------
    dataset : pd.Series
        Series containing image file paths
    labels : pd.Series
        Series containing encoded labels corresponding to each image
    augmentation_method : AugmentationMethod
        Data augmentation strategy applied to the training split only
    background_method : BackgroundMethod
        Background removal method applied during preprocessing
    test_size : float, optional
        Fraction of samples used as test set
    val_size : float, optional
        Fraction of samples used in the validation set
    input_size : int, optional
        Target image size, by default 224

    Returns
    -------
    tuple[PokemonDataset, PokemonDataset, PokemonDataset]
        The 3 resulting dataset, train/validation/test
    """
    x_temp, x_test, y_temp, y_test = train_test_split(
        dataset,
        labels,
        test_size=test_size,
        random_state=42,
        stratify=labels,
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_temp,
        y_temp,
        test_size=val_size / (1 - test_size),
        random_state=42,
        stratify=y_temp,
    )

    logger.info(
        f"Dataset split done ! (train={len(x_train)}, val={len(x_val)}, test={len(x_test)})",
    )

    x_train, y_train = data_augmentation(x_train, y_train, augmentation_method)

    logger.info(
        f"Dataset augmentation done ! (train={len(x_train)}, val={len(x_val)}, test={len(x_test)})",
    )

    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=np.array([0.485, 0.456, 0.406]),
                std=np.array([0.229, 0.224, 0.225]),
            ),
        ],
    )

    logger.info("Creating PyTorch dataset...")
    train_dataset = PokemonDataset(
        dataset=x_train,
        labels=y_train,
        background_method=background_method,
        transform=transform,
    )

    val_dataset = PokemonDataset(
        dataset=x_val,
        labels=y_val,
        background_method=background_method,
        transform=transform,
    )

    test_dataset = PokemonDataset(
        dataset=x_test,
        labels=y_test,
        background_method=background_method,
        transform=transform,
    )

    logger.info("PyTorch train/val/test datasets ready !")

    return train_dataset, val_dataset, test_dataset


@disable_logging(logging.INFO)
def evaluate_all_methods_dl(
    dataset: pd.Series,
    labels: pd.Series,
    label_encoder: LabelEncoder,
    models: list[BaseImageModel, dict | None],
    batch_size: int = 32,
    epochs: int = 10,
    lr: float = 1e-3,
    early_stopping_patience: int | None = 3,
) -> pd.DataFrame:
    """
    Evaluate all combinations of models, augmentation strategies and background removal methods

    Parameters
    ----------
    dataset : pd.Series
        Series containing image file paths
    labels : pd.Series
        Series containing the corresponding Pokemon labels
    label_encoder : LabelEncoder
        Fitted LabelEncoder used to retrieve class names for the
        classification report
    models : list[BaseImageModel, dict  |  None]
        List of tuples containing the model class to instantiate and the optional
        dictionary of initialization parameters

    Returns
    -------
    pd.DataFrame
        A dataframe containing the result for each configuration
    """
    results = []
    total = len(models) * len(AugmentationMethod) * len(BackgroundMethod)

    with tqdm(
        total=total,
        desc="Evaluation",
        bar_format="{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} {postfix}",
        colour="blue",
    ) as pbar:
        for model_class, model_params in models:
            for augmentation in AugmentationMethod:
                for background in BackgroundMethod:
                    pbar.set_postfix_str(
                        f"model={model_class.__name__} | "
                        f"augmentation={augmentation.name} | "
                        f"background={background.name}",
                    )

                    train_dataset, val_dataset, test_dataset = split_dataset_torch(
                        dataset,
                        labels,
                        augmentation,
                        background,
                    )

                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        generator=g,
                    )
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                    model: BaseImageModel = model_class(**model_params)

                    start = time.time()
                    model.fit(
                        train_loader=train_loader,
                        val_loader=val_loader,
                        epochs=epochs,
                        lr=lr,
                        early_stopping_patience=early_stopping_patience,
                    )
                    train_time = time.time() - start

                    y_pred = model.predict(test_loader)

                    y_true = []
                    for _, y_batch in test_loader:
                        y_true.extend(y_batch.numpy())

                    results.append(
                        {
                            "model": model_class.__name__,
                            "augmentation_method": augmentation.name,
                            "background_method": background.name,
                            "accuracy": accuracy_score(y_true, y_pred),
                            "precision_macro": precision_score(
                                y_true,
                                y_pred,
                                average="macro",
                                zero_division=0,
                            ),
                            "recall_macro": recall_score(
                                y_true,
                                y_pred,
                                average="macro",
                                zero_division=0,
                            ),
                            "f1_macro": f1_score(
                                y_true,
                                y_pred,
                                average="macro",
                                zero_division=0,
                            ),
                            "classification_report": classification_report(
                                y_true,
                                y_pred,
                                target_names=label_encoder.classes_,
                                zero_division=0,
                            ),
                            "train_time_sec": train_time,
                        },
                    )
                    pbar.update(1)

    return pd.DataFrame(results)
