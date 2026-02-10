import cv2
import numpy as np


def enhance_contrast_clahe(
    rgb: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> np.ndarray:
    """
    Enhance local contrast of an RGB image using CLAHE applied to the luminance channel in the Lab color space.
    (https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html)


    Parameters
    ----------
    rgb : np.ndarray
        Input RGB image
    clip_limit : float, optional
        CLAHE clipping limit
    tile_grid_size : int, optional
        Size of the grid used by CLAHE

    Returns
    -------
    np.ndarray
        RGB image with enhanced contrast
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    l_enhanced = clahe.apply(l_chan)

    lab_enhanced = cv2.merge((l_enhanced, a_chan, b_chan))
    bgr_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(bgr_enhanced, cv2.COLOR_BGR2RGB)


def preprocessing(rgb: np.ndarray) -> np.ndarray:
    """
    Apply the preprocessing pipeline to improve image quality

    Parameters
    ----------
    rgb : np.ndarray
        Input RGB image

    Returns
    -------
    np.ndarray
        Preprocessed RGB image
    """
    return enhance_contrast_clahe(rgb)
