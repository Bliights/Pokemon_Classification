from __future__ import annotations

import cv2
import numpy as np

from preprocessing import preprocessing


def _largest_component_bbox(
    bin_mask: np.ndarray,
    ignore_border: bool = True,
) -> tuple[int, int, int, int] | None:
    h, w = bin_mask.shape[:2]
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        bin_mask,
        connectivity=8,
    )
    if num <= 1:
        return None

    best: tuple[int, int, int, int] | None = None
    best_area = -1

    for i in range(1, num):
        x, y, bw, bh, area = stats[i]

        # Discard tiny connected components to reduce sensitivity to noise and texture.
        if area < 0.003 * h * w:
            continue

        # Optionally ignore components that touch the image border, which often correspond to background.
        if ignore_border and (x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1):
            continue

        # Track the largest valid component as the main candidate region.
        if area > best_area:
            best_area = area
            best = (x, y, bw, bh)

    return best


def bbox_by_saliency(
    im_bgr: np.ndarray,
    pad_ratio: float = 0.10,
    debug: bool = False,
) -> tuple[np.ndarray, tuple[int, int, int, int], dict[str, np.ndarray] | None]:
    h, w = im_bgr.shape[:2]

    # Apply a consistent preprocessing step to stabilize saliency estimation across inputs.
    pre = preprocessing(im_bgr)

    # Convert to grayscale and denoise to reduce high-frequency artifacts before computing saliency.
    gray = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Compute a saliency map using OpenCV's spectral residual method when available.
    sal_map: np.ndarray | None = None
    if hasattr(cv2, "saliency"):
        try:
            sal = cv2.saliency.StaticSaliencySpectralResidual_create()
            ok, sal_map = sal.computeSaliency(pre)
            if ok:
                sal_map = (sal_map * 255).astype(np.uint8)
        except Exception:
            sal_map = None

    # Fall back to a Laplacian-based response map if the saliency module is not present.
    if sal_map is None:
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        sal_map = cv2.normalize(
            np.abs(lap),
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        ).astype(np.uint8)

    # Binarize the saliency map using Otsu to isolate the most salient regions.
    _, th = cv2.threshold(
        sal_map,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # Clean the binary mask to remove small speckles and fill small holes.
    k = max(5, (min(h, w) // 80) | 1)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (k, k),
    )
    th = cv2.morphologyEx(
        th,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )
    th = cv2.morphologyEx(
        th,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    # Extract the bounding box of the largest salient connected component while avoiding border-connected regions.
    bbox = _largest_component_bbox(
        th,
        ignore_border=True,
    )

    # Relax border constraints if no suitable component was found under the stricter setting.
    if bbox is None:
        bbox = _largest_component_bbox(
            th,
            ignore_border=False,
        )

    # Use a centered crop as a last resort if saliency segmentation fails completely.
    if bbox is None:
        bbox = (
            int(0.15 * w),
            int(0.15 * h),
            int(0.70 * w),
            int(0.70 * h),
        )

    x, y, bw, bh = bbox

    # Prevent degenerate cases where the detected box covers almost the entire image.
    if bw * bh > 0.92 * w * h:
        bbox = (
            int(0.15 * w),
            int(0.15 * h),
            int(0.70 * w),
            int(0.70 * h),
        )
        x, y, bw, bh = bbox

    # Expand the box by a configurable padding ratio to include context around the salient region.
    px, py = int(pad_ratio * bw), int(pad_ratio * bh)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(w, x + bw + px), min(h, y + bh + py)

    # Apply a small additional expansion to improve robustness for tight saliency boxes.
    extra = 0.05
    box_w = x1 - x0
    box_h = y1 - y0
    ex = int(extra * box_w)
    ey = int(extra * box_h)

    x0 = max(0, x0 - ex)
    y0 = max(0, y0 - ey)
    x1 = min(w, x1 + ex)
    y1 = min(h, y1 + ey)

    crop = im_bgr[y0:y1, x0:x1].copy()

    if debug:
        # Provide intermediate artifacts and a visualization overlay to support inspection and tuning.
        vis = im_bgr.copy()
        cv2.rectangle(
            vis,
            (x0, y0),
            (x1, y1),
            (0, 255, 0),
            2,
        )
        dbg: dict[str, np.ndarray] = {
            "pre": pre,
            "sal": sal_map,
            "mask": th,
            "vis": vis,
        }
        return crop, (x0, y0, x1 - x0, y1 - y0), dbg

    return crop, (x0, y0, x1 - x0, y1 - y0), None
