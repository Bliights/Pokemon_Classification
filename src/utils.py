import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def build_dataframe(data_dir: Path) -> pd.DataFrame:
    """
    Build a pandas DataFrame using the fanart-dataset folder

    Parameters
    ----------
    data_dir : Path
        Directory containing the dataset images

    Returns
    -------
    pd.DataFrame
        A DataFrame with two columns:
            - `path`: absolute or relative path to the image file (as a string)
            - `pokemon`: extracted Pokémon label (as a string)
    """
    rows = []

    for img_path in data_dir.glob("*.jpg"):
        name = img_path.name.lower()

        # pattern: 000000__pokemon.jpg
        m = re.match(r"\d{6}__([a-z0-9_\-]+)\.jpg$", name)
        if not m:
            continue

        pokemon = m.group(1)

        rows.append(
            {
                "path": str(img_path),
                "pokemon": pokemon,
            },
        )

    df = pd.DataFrame(rows)

    return df.sort_values(["pokemon", "path"]).reset_index(drop=True)


def load_image(path: str | Path, *, as_gray: bool = False) -> np.ndarray:
    """
    Load an image from disk using OpenCV

    Parameters
    ----------
    path : str | Path
        Path to the image file
    as_gray : bool, optional
        If True, the image is loaded in grayscale, otherwise, it is loaded in color

    Returns
    -------
    np.ndarray
        Loaded image as a NumPy array

    Raises
    ------
    FileNotFoundError
        If the file does not exist or cannot be decoded
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Unable to load image: {path}")
    return img


def crop_with_bbox(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """
    Crop an image using a bounding box

    Parameters
    ----------
    image : np.ndarray
        Image (H, W, C)
    bbox : tuple[int, int, int, int]
        (x, y, width, height)

    Returns
    -------
    np.ndarray
        Cropped image
    """
    x, y, w, h = bbox
    return image[y : y + h, x : x + w].copy()
