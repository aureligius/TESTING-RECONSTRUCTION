"""
src/blob_detection.py

Ursula's blob-extraction approach: multi-scale Laplacian-of-Gaussian blob
enhancement, Otsu (with percentile fallback) thresholding, largest connected
component, intensity-weighted centroid.

NOTE on the "reflection at the edge, not the true tumor center" problem: this
is front-surface bias, not a bug in this code. The wave-facing near surface
of the tumor reflects first/strongest, so even the weighted centroid here is
still somewhat biased toward the tumor's front edge rather than its
geometric center — averaging over the whole blob mitigates this but doesn't
eliminate it. See metrics.py's front_surface_bias_diagnostic() for a check
that measures this directly instead of assuming it away.
"""

import numpy as np
from scipy.ndimage import label, gaussian_laplace


def select_blob_threshold_adaptive(img):
    """Otsu threshold if scikit-image is available, else 90th-percentile fallback."""
    try:
        from skimage.filters import threshold_otsu
        return threshold_otsu(img)
    except Exception:
        return np.percentile(img, 90)


def select_blob_mask_log(img, sigma_list=(2, 3, 4, 5)):
    """
    Multi-scale LoG blob enhancement before thresholding — suppresses
    high-frequency noise and enhances round, tumor-sized structures, so a
    single hot pixel from clutter is less likely to win over a genuine
    extended reflection.

    Returns (tumor_mask, blob_response, threshold_used).
    """
    try:
        response = np.zeros_like(img)
        for s in sigma_list:
            blob_r = -gaussian_laplace(img, sigma=s) * (s ** 2)
            response = np.maximum(response, blob_r)
        response = response / (response.max() + 1e-12)
        thresh = select_blob_threshold_adaptive(response)
        binary_mask = response >= thresh
    except Exception:
        response = img
        thresh = select_blob_threshold_adaptive(img)
        binary_mask = img >= thresh

    labeled_mask, n_blobs = label(binary_mask)
    if n_blobs > 0:
        sizes = [(labeled_mask == i).sum() for i in range(1, n_blobs + 1)]
        tumor_mask = labeled_mask == (np.argmax(sizes) + 1)
    else:
        tumor_mask = binary_mask
    return tumor_mask, response, thresh


def weighted_centroid(tumor_mask, enhanced_img, x_axis, y_axis):
    """Intensity-weighted centroid of the largest blob. Falls back to global
    argmax if no blob survived thresholding."""
    ys, xs = np.where(tumor_mask)
    if len(xs) > 0:
        weights = enhanced_img[ys, xs]
        peak_x = np.average(x_axis[xs], weights=weights)
        peak_y = np.average(y_axis[ys], weights=weights)
    else:
        fb = np.unravel_index(np.argmax(enhanced_img), enhanced_img.shape)
        peak_y, peak_x = y_axis[fb[0]], x_axis[fb[1]]
    return peak_x, peak_y


def extract_blob_candidate(img, x_axis, y_axis, use_log=True):
    """
    One-call blob extraction + localization. Returns a dict with tumor_mask,
    blob_response, threshold_used, peak_x, peak_y, plus features useful for
    the SVM verification stage later: blob_area_px, blob_compactness.
    """
    if use_log:
        tumor_mask, response, thresh = select_blob_mask_log(img)
    else:
        thresh = select_blob_threshold_adaptive(img)
        binary_mask = img >= thresh
        labeled_mask, n_blobs = label(binary_mask)
        if n_blobs > 0:
            sizes = [(labeled_mask == i).sum() for i in range(1, n_blobs + 1)]
            tumor_mask = labeled_mask == (np.argmax(sizes) + 1)
        else:
            tumor_mask = binary_mask
        response = img

    peak_x, peak_y = weighted_centroid(tumor_mask, img, x_axis, y_axis)

    area_px = int(tumor_mask.sum())
    if area_px > 0:
        # compactness: 4*pi*area / perimeter^2 (circle = 1, stringy shapes < 1)
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(tumor_mask)
        perimeter_px = max(int((tumor_mask & ~eroded).sum()), 1)
        compactness = (4 * np.pi * area_px) / (perimeter_px ** 2)
    else:
        compactness = 0.0

    return dict(
        tumor_mask=tumor_mask, blob_response=response, threshold_used=thresh,
        peak_x=peak_x, peak_y=peak_y, blob_area_px=area_px, blob_compactness=compactness,
    )