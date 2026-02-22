import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from config import DATA_AUGMENTATION_PATH, DATASET_PATH, AugmentationMethod
from logging_config import setup_logging

logger = logging.getLogger(__name__)
setup_logging(logging.INFO)


def flip_horizontal(img: Image.Image) -> Image.Image:
    """
    Apply a horizontal flip to the input image

    Parameters
    ----------
    img : Image.Image
        Input RGB image

    Returns
    -------
    Image.Image
        Horizontally flipped image
    """
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def rotate_plus_15(img: Image.Image) -> Image.Image:
    """
    Rotate the input image by +15 degrees

    Parameters
    ----------
    img : Image.Image
        Input RGB image

    Returns
    -------
    Image.Image
        Rotated image (+15 degrees)
    """
    return img.rotate(15, expand=True)


def rotate_minus_15(img: Image.Image) -> Image.Image:
    """
    Rotate the input image by -15 degrees

    Parameters
    ----------
    img : Image.Image
        Input RGB image

    Returns
    -------
    Image.Image
        Rotated image (-15 degrees)
    """
    return img.rotate(-15, expand=True)


def augment_dataset() -> None:
    """
    Generate deterministic augmented images for all images in DATASET_PATH

    For each input image, the following augmentations are applied:
        - Horizontal flip
        - Rotation +15°
        - Rotation -15°

    Augmented images are saved in DATA_AUGMENTATION_PATH using the
    original filename with a suffix indicating the transformation
    """
    input_dir = DATASET_PATH
    output_dir = DATA_AUGMENTATION_PATH
    augmentations = [
        ("flip", flip_horizontal),
        ("rot_p15", rotate_plus_15),
        ("rot_m15", rotate_minus_15),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)

    images = list(input_dir.glob("*.jpg"))
    total = len(images)

    saved = 0

    with tqdm(
        range(total),
        desc="Augmenting dataset",
        bar_format="{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} {postfix}",
        colour="green",
    ) as pbar:
        for i in pbar:
            img_path = images[i]

            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")

                    for suffix, transform in augmentations:
                        aug = transform(img)
                        new_name = f"{img_path.stem}_{suffix}.jpg"
                        aug.save(output_dir / new_name, "JPEG", quality=95)
                        saved += 1

            except Exception as e:
                logger.debug(f"Augmentation failed for {img_path} | {e}")

            pbar.set_postfix_str(f"saved={saved}")

    logger.info(
        "Augmentation finished ! (Original=%d | Generated=%d)",
        total,
        saved,
    )


def get_augmented_paths(img_path: Path) -> list[Path]:
    """
    Return all augmented image paths corresponding to a given original image

    Parameters
    ----------
    img_path : Path
        Path to the original image

    Returns
    -------
    list[Path]
        List of augmented image paths matching the original filename
    """
    stem = img_path.stem

    pattern = f"{stem}_*.jpg"

    return list(DATA_AUGMENTATION_PATH.glob(pattern))


def add_augmented_to_dataset(
    dataset: pd.Series,
    labels: np.ndarray,
) -> tuple[pd.Series, np.ndarray]:
    """
    Extend a dataset by adding augmented image paths while keeping the same labels

    Parameters
    ----------
    dataset : pd.Series
        Series of original image paths
    labels : np.ndarray
        Array of corresponding Pokémon labels

    Returns
    -------
    tuple[pd.Series, np.ndarray]
        Extended dataset paths and labels including augmented images
    """
    new_paths = []
    new_labels = []

    for img_path, label in zip(dataset, labels):
        img_path = Path(img_path)

        augmented_paths = get_augmented_paths(img_path)

        for aug_path in augmented_paths:
            new_paths.append(str(aug_path))
            new_labels.append(label)

    extended_dataset = pd.concat(
        [dataset, pd.Series(new_paths)],
        ignore_index=True,
    )

    extended_labels = np.concatenate(
        [labels, np.array(new_labels)],
        axis=0,
    )

    return extended_dataset, extended_labels


def data_augmentation(
    dataset: pd.Series,
    labels: np.ndarray,
    method: AugmentationMethod,
    **kwargs,
) -> tuple[pd.Series, np.ndarray]:
    """
    Data augmentation according to the method

    Parameters
    ----------
    dataset : pd.Series
        Series of original image paths
    labels : np.ndarray
        Array of corresponding Pokémon labels
    method : AugmentationMethod
        Augmentation type to extract

    Returns
    -------
    tuple[pd.Series, np.ndarray]
        Extended dataset paths and labels including augmented images

    Raises
    ------
    ValueError
        If "method" is not recognized
    """
    if method == AugmentationMethod.NONE:
        return dataset, labels
    if method == AugmentationMethod.TRUE:
        return add_augmented_to_dataset(dataset, labels, **kwargs)
    raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    augment_dataset()
