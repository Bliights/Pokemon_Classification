from pathlib import Path

import cv2
import numpy as np

from config import BACKGROUND_PATH, BackgroundMethod
from utils import load_image


def _largest_component(
    bin_mask: np.ndarray,
    ignore_border: bool = True,
) -> tuple[int, int, int, int] | None:
    """
    Return bbox of the largest connected component in a binary mask

    Parameters
    ----------
    bin_mask : np.ndarray
        Binary mask
    ignore_border : bool, optional
        Boolean to ignore component that touch the border

    Returns
    -------
    tuple[int, int, int, int] | None
        The bbox of the largest connected component
    """
    h, w = bin_mask.shape[:2]
    num, _, stats, _ = cv2.connectedComponentsWithStats(
        bin_mask,
        connectivity=8,
    )
    if num <= 1:
        return None

    best = None
    best_area = -1

    for i in range(1, num):
        x, y, bw, bh, area = stats[i]

        # Discard tiny connected components
        if area < 0.003 * h * w:
            continue

        # Discard border-touching connected components
        if ignore_border and (x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1):
            continue

        if area > best_area:
            best_area = area
            best = (x, y, bw, bh)

    return best


def segment_pokemon(im: np.ndarray, grabcut_iter: int = 5) -> np.ndarray:
    """
    Segment the main centered object using the GrabCut algorithm

    Parameters
    ----------
    im : np.ndarray
        Input BGR image
    grabcut_iter : int, optional
        Number of GrabCut refinement iterations

    Returns
    -------
    np.ndarray
        Binary mask
    """
    h, w = im.shape[:2]
    rect = (
        int(0.05 * w),
        int(0.05 * h),
        int(0.90 * w),
        int(0.90 * h),
    )

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        im,
        mask,
        rect,
        bgd_model,
        fgd_model,
        grabcut_iter,
        cv2.GC_INIT_WITH_RECT,
    )

    mask_bin = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype("uint8")

    # Cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel, iterations=1)

    bbox = _largest_component(mask_bin, ignore_border=False)
    if bbox is not None:
        x, y, bw, bh = bbox
        clean_mask = np.zeros_like(mask_bin)
        clean_mask[y : y + bh, x : x + bw] = mask_bin[y : y + bh, x : x + bw]
        mask_bin = clean_mask

    return mask_bin


def remove_background(
    im: np.ndarray,
    img_path: Path,
    method: BackgroundMethod,
    **kwargs,
) -> np.ndarray:
    """
    Methode to remove the background and keep only the pokemon

    Parameters
    ----------
    im : np.ndarray
        The original image
    img_path : Path
        Path to the original image
    method : BackgroundMethod
        Method of segmentation to use

    Returns
    -------
    np.ndarray
        Image with only the pokemon

    Raises
    ------
    ValueError
        Unknown method
    """
    if method == BackgroundMethod.NONE:
        return im
    if method == BackgroundMethod.GRABCUT:
        BACKGROUND_PATH.mkdir(parents=True, exist_ok=True)
        cached_path = BACKGROUND_PATH / f"{img_path.stem}_grabcut.jpg"
        if cached_path.exists():
            return load_image(cached_path)

        mask = segment_pokemon(im, **kwargs)
        segmented = cv2.bitwise_and(im, im, mask=mask)
        cv2.imencode(".jpg", segmented)[1].tofile(str(cached_path))
        return segmented
    raise ValueError(f"Unknown method: {method}")
