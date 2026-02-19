import cv2
import numpy as np


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


def compute_saliency_map(im_bgr: np.ndarray) -> np.ndarray:
    sal = cv2.saliency.StaticSaliencySpectralResidual_create()
    _, sal_map = sal.computeSaliency(im_bgr)

    return (sal_map * 255).astype(np.uint8)


def binarize_saliency(sal_map: np.ndarray) -> np.ndarray:
    h, w = sal_map.shape[:2]

    _, th = cv2.threshold(
        sal_map,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # Clean the binary mask
    k = max(5, (min(h, w) // 80) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)


def pad_bbox(
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    pad_ratio: float,
) -> tuple[int, int, int, int]:

    h, w = shape
    x, y, bw, bh = bbox

    px = int(pad_ratio * bw)
    py = int(pad_ratio * bh)

    x0 = max(0, x - px)
    y0 = max(0, y - py)
    x1 = min(w, x + bw + px)
    y1 = min(h, y + bh + py)

    return x0, y0, x1 - x0, y1 - y0


def bbox_by_saliency(
    im_bgr: np.ndarray,
    pad_ratio: float = 0.10,
) -> tuple[int, int, int, int]:

    h, w = im_bgr.shape[:2]

    sal_map = compute_saliency_map(im_bgr)
    mask = binarize_saliency(sal_map)

    bbox = _largest_component(mask, ignore_border=True)

    if bbox is None:
        return (0, 0, w, h)

    return sal_map, mask, pad_bbox(bbox, (h, w), pad_ratio)
