import cv2
import numpy as np

from config import ExtractMethod


def extract_sift_descriptors(
    img: np.ndarray,
    *,
    max_features: int = 800,
) -> np.ndarray:
    """
    Extract SIFT local descriptors from an image

    Parameters
    ----------
    img : np.ndarray
        Input image
    max_features : int, optional
        Maximum number of keypoints/descriptors to keep

    Returns
    -------
    np.ndarray
        Descriptor matrix
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=max_features)
    _, desc = sift.detectAndCompute(gray, None)
    if desc is None:
        return np.empty((0, 0), dtype=np.float32)
    return desc.astype(np.float32)


def extract_orb_descriptors(
    img: np.ndarray,
    *,
    max_features: int = 800,
) -> np.ndarray:
    """
    Extract ORB local descriptors from an image

    Parameters
    ----------
    img : np.ndarray
        Input image
    max_features : int, optional
        Maximum number of keypoints/descriptors to keep

    Returns
    -------
    np.ndarray
        Descriptor matrix
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=max_features)
    _, desc = orb.detectAndCompute(gray, None)
    if desc is None:
        return np.empty((0, 0), dtype=np.float32)
    return desc.astype(np.float32)


def extract_descriptors(
    img: np.ndarray,
    method: ExtractMethod,
    **kwargs,
) -> np.ndarray:
    """
    Descriptor extraction according to the selected method

    Parameters
    ----------
    img : np.ndarray
        Input image
    method : ExtractMethod
        Descriptor type to extract

    Returns
    -------
    np.ndarray
        Descriptor matrix

    Raises
    ------
    ValueError
        If "method" is not recognized
    """
    if method == ExtractMethod.SIFT:
        return extract_sift_descriptors(img, **kwargs)
    if method == ExtractMethod.ORB:
        return extract_orb_descriptors(img, **kwargs)
    raise ValueError(f"Unknown method: {method}")
