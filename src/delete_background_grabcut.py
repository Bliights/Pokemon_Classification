from __future__ import annotations

import cv2
import numpy as np

from bbox import bbox_by_saliency


def remove_background_inside_bbox(
    im_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    th_mask_full: np.ndarray,
    gc_iters: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x0, y0, w, h = bbox
    height, width = im_bgr.shape[:2]

    roi = im_bgr[y0 : y0 + h, x0 : x0 + w].copy()
    th_roi = th_mask_full[y0 : y0 + h, x0 : x0 + w].copy()

    # Initialize GrabCut labels (BG, FG, probable BG, probable FG) from a prior mask.
    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)

    border = max(2, int(0.04 * min(h, w)))
    k = max(3, (min(h, w) // 60) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # Force background on a thin border to stabilize GrabCut and avoid edge leakage.
    gc[:border, :] = cv2.GC_BGD
    gc[-border:, :] = cv2.GC_BGD
    gc[:, :border] = cv2.GC_BGD
    gc[:, -border:] = cv2.GC_BGD

    # Seed probable foreground from the binary saliency mask when available.
    if np.count_nonzero(th_roi) > 0:
        gc[th_roi > 0] = cv2.GC_PR_FGD

    # Define a "sure foreground" seed; prefer the provided mask, otherwise fall back to a central ellipse.
    sure_fg = th_roi.copy() if np.count_nonzero(th_roi) > 0 else None

    if sure_fg is None or np.count_nonzero(sure_fg) == 0:
        sure_fg = np.zeros((h, w), np.uint8)
        cx, cy = w // 2, h // 2
        ax = max(5, int(0.28 * w))
        ay = max(5, int(0.28 * h))
        cv2.ellipse(sure_fg, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)

    gc[sure_fg > 0] = cv2.GC_FGD

    # Ensure both background and foreground classes exist; otherwise reinitialize to a safe configuration.
    has_bg = np.any((gc == cv2.GC_BGD) | (gc == cv2.GC_PR_BGD))
    has_fg = np.any((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD))
    if not has_bg or not has_fg:
        gc[:] = cv2.GC_PR_BGD
        gc[:border, :] = cv2.GC_BGD
        gc[-border:, :] = cv2.GC_BGD
        gc[:, :border] = cv2.GC_BGD
        gc[:, -border:] = cv2.GC_BGD
        gc[sure_fg > 0] = cv2.GC_FGD

    # Run GrabCut using the prepared label mask.
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(
        roi,
        gc,
        None,
        bgd_model,
        fgd_model,
        gc_iters,
        cv2.GC_INIT_WITH_MASK,
    )

    # Convert GrabCut labels into a binary foreground mask for the ROI.
    fg_roi = np.where(
        (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)

    # Remove any region connected to the ROI border to eliminate background leakage inside the bounding box.
    obj = (fg_roi > 0).astype(np.uint8) * 255
    inv = cv2.bitwise_not(obj)

    ff = inv.copy()
    roi_h, roi_w = ff.shape
    mask_ff = np.zeros((roi_h + 2, roi_w + 2), np.uint8)

    cv2.floodFill(ff, mask_ff, (0, 0), 255)
    cv2.floodFill(ff, mask_ff, (roi_w - 1, 0), 255)
    cv2.floodFill(ff, mask_ff, (0, roi_h - 1), 255)
    cv2.floodFill(ff, mask_ff, (roi_w - 1, roi_h - 1), 255)

    keep = cv2.bitwise_not(ff)

    # Keep only the largest connected component after border cleanup to focus on the main subject.
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (keep > 0).astype(np.uint8),
        connectivity=8,
    )
    if num > 1:
        areas = stats[1:, 4]
        idx = 1 + int(np.argmax(areas))
        keep = np.where(labels == idx, 255, 0).astype(np.uint8)
    else:
        keep = obj

    # Slightly dilate to recover thin contours that might be lost during cleanup.
    keep = cv2.dilate(keep, kernel, iterations=1)

    fg_roi = keep

    # Reproject the ROI mask back to full image coordinates.
    fg_mask_full = np.zeros((height, width), np.uint8)
    fg_mask_full[y0 : y0 + h, x0 : x0 + w] = fg_roi

    cutout_bgr = cv2.bitwise_and(im_bgr, im_bgr, mask=fg_mask_full)

    cutout_rgba = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2BGRA)
    cutout_rgba[:, :, 3] = fg_mask_full

    return fg_mask_full, cutout_bgr, cutout_rgba


def bbox_and_cutout(
    im_bgr: np.ndarray,
    pad_ratio: float = 0.10,
    gc_iters: int = 5,
    debug: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[int, int, int, int],
    dict[str, np.ndarray] | None,
]:
    # Compute a saliency-driven bounding box and return intermediate debug artifacts.
    crop, bbox, dbg = bbox_by_saliency(
        im_bgr,
        pad_ratio=pad_ratio,
        debug=True,
    )

    x0, y0, w, h = bbox

    # Use the saliency mask as a prior to refine segmentation within the bounding box.
    th_full = dbg["mask"]
    fg_mask_full, cutout_bgr, cutout_rgba = remove_background_inside_bbox(
        im_bgr,
        (x0, y0, w, h),
        th_full,
        gc_iters=gc_iters,
    )

    # Produce cropped outputs aligned with the detected bounding box.
    cutout_crop_bgr = cutout_bgr[y0 : y0 + h, x0 : x0 + w].copy()
    cutout_crop_rgba = cutout_rgba[y0 : y0 + h, x0 : x0 + w].copy()

    if debug:
        dbg2: dict[str, np.ndarray] = dict(dbg)
        dbg2["fg_mask_full"] = fg_mask_full
        dbg2["cutout_crop_bgr"] = cutout_crop_bgr
        return cutout_crop_bgr, cutout_crop_rgba, bbox, dbg2

    return cutout_crop_bgr, cutout_crop_rgba, bbox, None
