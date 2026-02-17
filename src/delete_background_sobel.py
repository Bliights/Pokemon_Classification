from __future__ import annotations

import cv2
import numpy as np


def remove_background_inside_bbox_sobel(
    im_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    pad_mask_dilate: int = 1,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray] | None]:
    """
    Remove the background within a given bounding box using an edge-based baseline.

    This method avoids GrabCut and relies on:
    Sobel magnitude -> Otsu threshold -> morphological cleanup -> flood-fill border removal
    -> keep largest connected component -> build cutouts (BGR and RGBA).

    Parameters
    ----------
    im_bgr:
        Input image in BGR format.
    bbox:
        Bounding box as (x0, y0, w, h) in full-image coordinates.
    pad_mask_dilate:
        Optional dilation iterations applied to the final ROI mask to preserve thin contours.
    debug:
        If True, returns intermediate processing images for inspection.

    Returns
    -------
    fg_mask_full:
        Full-size uint8 mask (0/255) of the extracted foreground.
    cutout_bgr:
        Full-size BGR cutout with background set to black.
    cutout_rgba:
        Full-size BGRA cutout with alpha channel set from the foreground mask.
    dbg:
        Optional dictionary of debug images; None when debug is False.
    """
    x0, y0, w, h = bbox
    height, width = im_bgr.shape[:2]

    # Extract ROI for local processing while keeping original full-image coordinates.
    roi = im_bgr[y0 : y0 + h, x0 : x0 + w].copy()
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Compute Sobel gradient magnitude to emphasize object boundaries.
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(
        mag,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)

    # Convert the gradient magnitude to a binary edge mask using Otsu thresholding.
    _, edges = cv2.threshold(
        mag,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # Close gaps and thicken edges so the interior becomes a contiguous region.
    k = max(3, (min(h, w) // 70) | 1)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (k, k),
    )
    closed = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )
    filled = cv2.dilate(
        closed,
        kernel,
        iterations=1,
    )

    # Remove any region connected to the ROI border by flood-filling the background on the inverted mask.
    obj = (filled > 0).astype(np.uint8) * 255
    inv = cv2.bitwise_not(obj)
    ff = inv.copy()

    mask_ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, mask_ff, (0, 0), 255)
    cv2.floodFill(ff, mask_ff, (w - 1, 0), 255)
    cv2.floodFill(ff, mask_ff, (0, h - 1), 255)
    cv2.floodFill(ff, mask_ff, (w - 1, h - 1), 255)

    keep = cv2.bitwise_not(ff)

    # Select the largest connected component as the primary foreground candidate.
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (keep > 0).astype(np.uint8),
        connectivity=8,
    )
    if num > 1:
        areas = stats[1:, 4]
        idx = 1 + int(np.argmax(areas))
        keep = np.where(labels == idx, 255, 0).astype(np.uint8)
    else:
        keep = np.zeros((h, w), np.uint8)

    # Optionally expand the mask slightly to preserve thin contours around the subject.
    if pad_mask_dilate > 0:
        keep = cv2.dilate(
            keep,
            kernel,
            iterations=int(pad_mask_dilate),
        )

    # Reproject the ROI mask into a full-size mask aligned with the original image.
    fg_mask_full = np.zeros((height, width), np.uint8)
    fg_mask_full[y0 : y0 + h, x0 : x0 + w] = keep

    # Generate cutouts from the full-size mask (black background for BGR, alpha for RGBA).
    cutout_bgr = cv2.bitwise_and(
        im_bgr,
        im_bgr,
        mask=fg_mask_full,
    )
    cutout_rgba = cv2.cvtColor(
        im_bgr,
        cv2.COLOR_BGR2BGRA,
    )
    cutout_rgba[:, :, 3] = fg_mask_full

    dbg: dict[str, np.ndarray] | None = None
    if debug:
        dbg = {
            "mag": mag,
            "edges": edges,
            "closed": closed,
            "keep_roi": keep,
        }

    return fg_mask_full, cutout_bgr, cutout_rgba, dbg
