"""
baseline_argmax_test.py

Standalone isolation test of Aurel's ORIGINAL physics + localization
approach, run across all valid scans (not just A16F14). Does NOT touch any
existing module — reuses physics.py / beamforming.py / signal_processing.py
/ data_loading.py exactly as they exist.

Deliberately different from pipeline.py / ablation_runner.py in two ways:
  1. Localization: global ARGMAX (brightest pixel), matching Aurel's
     original recon_utils.compute_metrics() — NOT Ursula's blob-centroid.
  2. SCR: fixed 2mm ROI around ground truth (matching Aurel's original
     default), NOT the phantom's actual tumor radius.

Variants: Raw DAS, Raw DMAS, DAS+CF, DMAS+CF. No bent-ray, no TVSVD.
Grid: 0.25mm spacing, breast_radius * 1.5 margin (matches Aurel's original
sizing). No images saved or displayed — printed summary only.

Usage:
    python baseline_argmax_test.py --n-scans 20   # smoke test
    python baseline_argmax_test.py                # full run, all valid scans
"""

import argparse
import time as time_module

import numpy as np
import pandas as pd
from scipy import ndimage
from joblib import Parallel, delayed

from src.data_loading import load_all_data
from select_phantoms import build_phantom_table, stratified_select
from src import physics
from src import signal_processing as sp
from src import beamforming as bf

GRID_STEP_MM = 0.25
GRID_MARGIN_FACTOR = 1.5
SCR_ROI_RADIUS_MM = 2.0
DETECTION_THRESHOLD_MM = 20.0
CENTER_EXCLUDE_RADIUS_MM = 5.0   # radius around the geometric center (0,0) to
                                  # mask out before argmax — the antenna ring's
                                  # rotational symmetry makes this pixel
                                  # artificially bright/coherent regardless of
                                  # any real tumor (see chat discussion)

CF_BASELINE_N_TRIALS = 10
CF_BASELINE_SEED = 12345
_CF_BASELINE_CACHE = {}   # phant_id -> (baseline_mean, baseline_std)


def compute_cf_baseline(delay_grid, time_axis, n_ant, n_trials=CF_BASELINE_N_TRIALS,
                         seed=CF_BASELINE_SEED):
    """
    Runs das_coherent_cf on N_TRIALS of pure independent random complex noise
    (no real signal, no ICZT, no calibration — just random values per
    antenna), through the SAME delay_grid used for the real scan. Since CF's
    formula is scale-invariant, noise amplitude doesn't matter — only its
    lack of genuine cross-antenna structure does. This isolates exactly how
    much CF is inflated by antenna-ring geometry ALONE, independent of any
    real signal, at every pixel.

    Returns (baseline_mean, baseline_std) — per-pixel mean and std of CF
    across the N_TRIALS noise-only runs.
    """
    rng = np.random.default_rng(seed)
    n_time = len(time_axis)
    cf_maps = []
    for _ in range(n_trials):
        noise_signal = (rng.standard_normal((n_time, n_ant))
                         + 1j * rng.standard_normal((n_time, n_ant)))
        _, cf_map, _ = bf.das_coherent_cf(noise_signal, time_axis, delay_grid)
        cf_maps.append(cf_map)
    cf_stack = np.stack(cf_maps, axis=0)
    return cf_stack.mean(axis=0), cf_stack.std(axis=0)


def get_cf_baseline_cached(phant_id, delay_grid, time_axis, n_ant):
    """Baseline only depends on phantom geometry (delay_grid), not the
    actual scan/tumor — cache per phant_id so repeated phantoms in a sample
    don't recompute it."""
    if phant_id not in _CF_BASELINE_CACHE:
        _CF_BASELINE_CACHE[phant_id] = compute_cf_baseline(delay_grid, time_axis, n_ant)
    return _CF_BASELINE_CACHE[phant_id]


def debias_cf_subtraction(cf_map, baseline_mean):
    return np.clip(cf_map - baseline_mean, 0, None)


def debias_cf_ratio(cf_map, baseline_mean, eps=1e-6):
    return cf_map / (baseline_mean + eps)


def debias_cf_zscore(cf_map, baseline_mean, baseline_std, eps=1e-6):
    z = (cf_map - baseline_mean) / (baseline_std + eps)
    return np.clip(z, 0, None)   # negative z (below chance-level coherence) -> 0 weight


TOP_K_CANDIDATES = 5
CANDIDATE_MIN_SEPARATION_MM = 10.0   # minimum spacing between candidate peaks
CANDIDATE_LOCAL_WINDOW_MM = 3.0       # window radius for the area/compactness feature
CANDIDATE_LABEL_THRESHOLD_MM = DETECTION_THRESHOLD_MM   # reuse the 20mm "detection" convention


def find_top_k_peaks(img, axis_mm, k=TOP_K_CANDIDATES, min_separation_mm=CANDIDATE_MIN_SEPARATION_MM):
    """Greedy non-max-suppression peak finder: repeatedly take the brightest
    remaining pixel, mask out a min_separation_mm radius around it, repeat k
    times. Returns a list of (px_mm, py_mm, iy, ix) tuples, brightest first."""
    step_mm = axis_mm[1] - axis_mm[0]
    sep_px = max(1, int(round(min_separation_mm / step_mm)))
    search_img = img.copy()
    peaks = []
    for _ in range(k):
        iy, ix = np.unravel_index(np.argmax(search_img), search_img.shape)
        if not np.isfinite(search_img[iy, ix]):
            break
        px, py = axis_mm[ix], axis_mm[iy]
        peaks.append((px, py, iy, ix))
        y0, y1 = max(0, iy - sep_px), min(search_img.shape[0], iy + sep_px + 1)
        x0, x1 = max(0, ix - sep_px), min(search_img.shape[1], ix + sep_px + 1)
        search_img[y0:y1, x0:x1] = -np.inf
    return peaks


HESSIAN_STEP_MM = 1.0   # finite-difference step, matches expected real-tumor blob scale
                         # (rather than pixel-level noise at the 0.25mm grid resolution)


def compute_hessian_eigenratio(img, axis_mm, iy, ix, step_mm=HESSIAN_STEP_MM):
    """
    Hessian (2nd-derivative) eigenvalue ratio at a candidate peak — the
    'blobness' feature suggested by both external reviews and confirmed as
    top-priority by our own AUC analysis (see chat discussion: CF-based
    features are weak/wrong-direction, geometry helps, shape is untested).

    A round, compact reflector (real tumor) curves down similarly in every
    direction -> eigenvalues of similar magnitude -> ratio near 1.
    An elongated ridge (grating-lobe artifact) curves sharply in one
    direction, gently in the other -> ratio near 0.

    Uses a step size matched to a physical scale (~1mm) rather than single
    adjacent pixels, since single-pixel finite differences are noisy at
    0.25mm grid resolution.
    """
    step_mm_actual = axis_mm[1] - axis_mm[0]
    step_px = max(1, int(round(step_mm / step_mm_actual)))
    ny, nx = img.shape

    iy_p, iy_m = min(iy + step_px, ny - 1), max(iy - step_px, 0)
    ix_p, ix_m = min(ix + step_px, nx - 1), max(ix - step_px, 0)

    if iy_p == iy or iy_m == iy or ix_p == ix or ix_m == ix:
        return 0.0   # candidate too close to grid edge for a meaningful stencil

    center_val = img[iy, ix]
    ixx = img[iy, ix_p] - 2 * center_val + img[iy, ix_m]
    iyy = img[iy_p, ix] - 2 * center_val + img[iy_m, ix]
    ixy = (img[iy_p, ix_p] - img[iy_p, ix_m] - img[iy_m, ix_p] + img[iy_m, ix_m]) / 4.0

    hessian = np.array([[ixx, ixy], [ixy, iyy]])
    eigvals = np.abs(np.linalg.eigvalsh(hessian))
    max_eig, min_eig = eigvals.max(), eigvals.min()
    if max_eig < 1e-12:
        return 0.0
    return float(min_eig / max_eig)


def extract_candidate_features(img, cf_map, axis_mm, iy, ix, px_mm, py_mm,
                                shell_center_mm, birads, breast_radius_mm,
                                window_mm=CANDIDATE_LOCAL_WINDOW_MM,
                                scr_roi_radius_mm=SCR_ROI_RADIUS_MM):
    """9-feature vector for one candidate peak: CF-at-peak, CF-to-background
    ratio, local area/compactness (self-contained — NOT Ursula's Otsu/LoG
    blob detection), distance to shell center, distance to antenna-ring
    geometric center (absolute mm AND as a ratio of the phantom's own
    radius), local SCR, Hessian eigenvalue ratio (blobness), BI-RADS class.

    dist_to_ring_center_ratio added after confirming (see chat discussion)
    that SCR collapses specifically once a tumor crosses ~90% of the way
    from center to the phantom boundary — a size-relative measure, not an
    absolute mm one, since a fixed mm distance means something different in
    a 22.9mm-radius phantom (A1) than a 57mm-radius one (A16).

    hessian_eigenratio added after AUC analysis showed CF-based features
    are weak or wrong-direction (see chat discussion) — this targets shape,
    orthogonal to brightness/coherence."""
    cf_at_peak = float(cf_map[iy, ix])
    flat_idx = iy * cf_map.shape[1] + ix
    cf_background_mean = float(np.mean(np.delete(cf_map.ravel(), flat_idx)))
    cf_background_ratio = cf_at_peak / (cf_background_mean + 1e-10)

    step_mm = axis_mm[1] - axis_mm[0]
    win_px = max(1, int(round(window_mm / step_mm)))
    y0, y1 = max(0, iy - win_px), min(img.shape[0], iy + win_px + 1)
    x0, x1 = max(0, ix - win_px), min(img.shape[1], ix + win_px + 1)
    window = img[y0:y1, x0:x1]
    peak_val = img[iy, ix]
    area_fraction = float(np.mean(window >= 0.5 * peak_val)) if peak_val > 0 else 0.0

    dist_to_shell_center = float(np.hypot(px_mm - shell_center_mm[0], py_mm - shell_center_mm[1]))
    dist_to_ring_center = float(np.hypot(px_mm, py_mm))
    dist_to_ring_center_ratio = dist_to_ring_center / breast_radius_mm if breast_radius_mm > 0 else 0.0

    gx_mm, gy_mm = np.meshgrid(axis_mm, axis_mm)
    roi_mask = np.sqrt((gx_mm - px_mm) ** 2 + (gy_mm - py_mm) ** 2) <= scr_roi_radius_mm
    outside_mean = img[~roi_mask].mean() if (~roi_mask).any() else 1e-10
    local_scr = float(20 * np.log10(np.abs(peak_val) / (np.abs(outside_mean) + 1e-10))) \
        if peak_val > 0 and outside_mean > 0 else 0.0

    hessian_eigenratio = compute_hessian_eigenratio(img, axis_mm, iy, ix)

    return dict(
        cf_at_peak=cf_at_peak,
        cf_background_ratio=cf_background_ratio,
        area_fraction=area_fraction,
        dist_to_shell_center_mm=dist_to_shell_center,
        dist_to_ring_center_mm=dist_to_ring_center,
        dist_to_ring_center_ratio=dist_to_ring_center_ratio,
        local_scr_db=local_scr,
        hessian_eigenratio=hessian_eigenratio,
        birads=birads,
    )


def generate_candidates(img, cf_map, axis_mm, gt_x_mm, gt_y_mm, shell_center_mm, birads,
                         breast_radius_mm, phant_id,
                         k=TOP_K_CANDIDATES, label_threshold_mm=CANDIDATE_LABEL_THRESHOLD_MM,
                         artifact_exclusion_radius_mm=CENTER_EXCLUDE_RADIUS_MM):
    """Top-K candidates + features + label (1 = within label_threshold_mm of
    ground truth, else 0) for one image.

    artifact_exclusion_radius_mm: candidates within this radius of the
    antenna-ring geometric center are ALWAYS labeled 0, even if they happen
    to fall within label_threshold_mm of ground truth. Without this, the
    confirmed ring-symmetry artifact (see chat discussion — CF~0.8 at
    center vs ~0.03 elsewhere, present in every scan regardless of tumor
    position) gets labeled 1 whenever the true tumor coincidentally lands
    near center, contaminating the "real tumor" feature distribution with
    non-detecting artifact candidates that happened to get lucky."""
    peaks = find_top_k_peaks(img, axis_mm, k=k)
    candidates = []
    for px_mm, py_mm, iy, ix in peaks:
        features = extract_candidate_features(img, cf_map, axis_mm, iy, ix, px_mm, py_mm,
                                               shell_center_mm, birads, breast_radius_mm)
        le_to_gt = float(np.hypot(px_mm - gt_x_mm, py_mm - gt_y_mm))
        # Signed residuals (not just unsigned distance) — used for the axis-
        # orientation/origin check (see chat discussion): a systematic axis
        # swap or sign flip would show up as a consistent directional bias
        # in these residuals across many label=1 (correctly matched)
        # candidates, rather than scattering randomly around zero.
        residual_x_mm = float(px_mm - gt_x_mm)
        residual_y_mm = float(py_mm - gt_y_mm)
        dist_to_ring_center = features["dist_to_ring_center_mm"]
        if dist_to_ring_center <= artifact_exclusion_radius_mm:
            label = 0   # known artifact zone — never a genuine detection
        else:
            label = 1 if le_to_gt <= label_threshold_mm else 0
        candidates.append(dict(phant_id=phant_id, peak_x_mm=px_mm, peak_y_mm=py_mm,
                                le_to_gt_mm=le_to_gt,
                                residual_x_mm=residual_x_mm, residual_y_mm=residual_y_mm,
                                label=label, **features))
    return candidates


def build_grid(breast_radius_mm, margin_factor=GRID_MARGIN_FACTOR, step_mm=GRID_STEP_MM):
    grid_radius_mm = breast_radius_mm * margin_factor
    axis_mm = np.arange(-grid_radius_mm, grid_radius_mm + step_mm, step_mm)
    grid_x_mm, grid_y_mm = np.meshgrid(axis_mm, axis_mm)
    return grid_x_mm, grid_y_mm, axis_mm, grid_radius_mm


def argmax_localize_and_score(img, axis_mm, grid_radius_mm, gt_x_mm, gt_y_mm,
                               roi_radius_mm=SCR_ROI_RADIUS_MM,
                               exclude_center_radius_mm=None):
    """Global argmax peak + LE + fixed-ROI SCR, matching Aurel's original
    compute_metrics() exactly (20*log10, not 10*log10; fixed 2mm ROI, not
    the phantom's actual tumor radius).

    exclude_center_radius_mm: if set, pixels within this radius of (0,0) are
    excluded from the argmax search (set to -inf before argmax) — masks the
    antenna-ring geometric-center artifact. SCR's tumor/clutter ROI is
    unaffected — only affects where the peak search looks."""
    search_img = img
    if exclude_center_radius_mm is not None:
        gx_mm, gy_mm = np.meshgrid(axis_mm, axis_mm)
        center_dist = np.sqrt(gx_mm ** 2 + gy_mm ** 2)
        search_img = np.where(center_dist <= exclude_center_radius_mm, -np.inf, img)

    peak = np.unravel_index(np.argmax(search_img), search_img.shape)
    px = axis_mm[peak[1]]
    py = axis_mm[peak[0]]
    le = np.sqrt((px - gt_x_mm) ** 2 + (py - gt_y_mm) ** 2)

    gx_mm, gy_mm = np.meshgrid(axis_mm, axis_mm)
    tumor_dist = np.sqrt((gx_mm - gt_x_mm) ** 2 + (gy_mm - gt_y_mm) ** 2)
    tumor_mask = tumor_dist <= roi_radius_mm
    clutter_mask = ~tumor_mask

    tumor_peak = img[tumor_mask].max() if tumor_mask.any() else 1e-10
    clutter_mean = img[clutter_mask].mean() if clutter_mask.any() else 1e-10

    if clutter_mean <= 0 or tumor_peak <= 0:
        scr = 0.0
    else:
        scr = 20 * np.log10(np.abs(tumor_peak) / (np.abs(clutter_mean) + 1e-10))

    on_edge = (abs(abs(px) - grid_radius_mm) < 1e-6) or (abs(abs(py) - grid_radius_mm) < 1e-6)
    return le, scr, on_edge, px, py


def weighted_centroid_localize_and_score(img, axis_mm, grid_radius_mm, gt_x_mm, gt_y_mm,
                                          roi_radius_mm=SCR_ROI_RADIUS_MM,
                                          threshold_fraction=0.5,
                                          exclude_center_radius_mm=None):
    """
    Alternative to pure argmax: find the peak (optionally center-excluded,
    same as argmax_localize_and_score), threshold the image at
    threshold_fraction of the peak's value, isolate the connected blob
    containing that peak (scipy.ndimage.label — standard library, NOT
    Ursula's Otsu/LoG pipeline), then take the intensity-weighted centroid
    of that whole blob rather than trusting the single brightest pixel.

    Tests whether "brightest single pixel" itself is the bottleneck for LE,
    independent of the CF-debiasing question (see chat discussion).
    """
    search_img = img
    if exclude_center_radius_mm is not None:
        gx_mm, gy_mm = np.meshgrid(axis_mm, axis_mm)
        center_dist = np.sqrt(gx_mm ** 2 + gy_mm ** 2)
        search_img = np.where(center_dist <= exclude_center_radius_mm, -np.inf, img)

    peak = np.unravel_index(np.argmax(search_img), search_img.shape)
    peak_val = img[peak]

    threshold = threshold_fraction * peak_val
    binary_mask = img >= threshold
    labeled_mask, n_blobs = ndimage.label(binary_mask)
    blob_label = labeled_mask[peak]

    if blob_label == 0 or n_blobs == 0:
        # peak itself didn't survive its own threshold (shouldn't normally
        # happen) — fall back to the raw peak location
        px, py = axis_mm[peak[1]], axis_mm[peak[0]]
    else:
        blob_mask = labeled_mask == blob_label
        ys, xs = np.where(blob_mask)
        weights = img[ys, xs]
        px = float(np.average(axis_mm[xs], weights=weights))
        py = float(np.average(axis_mm[ys], weights=weights))

    le = float(np.sqrt((px - gt_x_mm) ** 2 + (py - gt_y_mm) ** 2))

    gx_mm, gy_mm = np.meshgrid(axis_mm, axis_mm)
    tumor_dist = np.sqrt((gx_mm - gt_x_mm) ** 2 + (gy_mm - gt_y_mm) ** 2)
    tumor_mask = tumor_dist <= roi_radius_mm
    clutter_mask = ~tumor_mask

    tumor_peak = img[tumor_mask].max() if tumor_mask.any() else 1e-10
    clutter_mean = img[clutter_mask].mean() if clutter_mask.any() else 1e-10

    if clutter_mean <= 0 or tumor_peak <= 0:
        scr = 0.0
    else:
        scr = float(20 * np.log10(np.abs(tumor_peak) / (np.abs(clutter_mean) + 1e-10)))

    on_edge = (abs(abs(px) - grid_radius_mm) < 1e-6) or (abs(abs(py) - grid_radius_mm) < 1e-6)
    return le, scr, on_edge, px, py


def largest_blob_localize_and_score(img, axis_mm, grid_radius_mm, gt_x_mm, gt_y_mm,
                                     roi_radius_mm=SCR_ROI_RADIUS_MM,
                                     percentile_threshold=90.0):
    """
    Largest-connected-AREA localization — the specific mechanism from
    Ursula's blob-detection approach we hadn't tested (self-contained
    reimplementation, not her actual module — see chat discussion on
    keeping this script isolated from her contributions).

    KEY DIFFERENCE from weighted_centroid_localize_and_score: that function
    always picks the blob CONTAINING THE PEAK — if the peak is the
    ring-center artifact, it's structurally trapped in the artifact's blob
    every time. This function thresholds the WHOLE image globally (90th
    percentile, not relative to any one candidate pixel) and picks whichever
    connected region has the most PIXELS, regardless of whether it contains
    the single brightest point. If the center artifact is a small, sharp,
    concentrated spot (plausible — perfect symmetry tends to create a
    narrow peak) while a real tumor's reflection spreads over a larger
    area even if dimmer, this method could select the tumor's blob instead
    — something the peak-anchored version can never do by construction.
    """
    threshold = np.percentile(img, percentile_threshold)
    binary_mask = img >= threshold
    labeled_mask, n_blobs = ndimage.label(binary_mask)

    if n_blobs == 0:
        peak = np.unravel_index(np.argmax(img), img.shape)
        px, py = axis_mm[peak[1]], axis_mm[peak[0]]
    else:
        sizes = ndimage.sum(binary_mask, labeled_mask, range(1, n_blobs + 1))
        largest_label = int(np.argmax(sizes)) + 1
        blob_mask = labeled_mask == largest_label
        ys, xs = np.where(blob_mask)
        weights = img[ys, xs]
        px = float(np.average(axis_mm[xs], weights=weights))
        py = float(np.average(axis_mm[ys], weights=weights))

    le = float(np.sqrt((px - gt_x_mm) ** 2 + (py - gt_y_mm) ** 2))

    gx_mm, gy_mm = np.meshgrid(axis_mm, axis_mm)
    tumor_dist = np.sqrt((gx_mm - gt_x_mm) ** 2 + (gy_mm - gt_y_mm) ** 2)
    tumor_mask = tumor_dist <= roi_radius_mm
    clutter_mask = ~tumor_mask

    tumor_peak = img[tumor_mask].max() if tumor_mask.any() else 1e-10
    clutter_mean = img[clutter_mask].mean() if clutter_mask.any() else 1e-10

    if clutter_mean <= 0 or tumor_peak <= 0:
        scr = 0.0
    else:
        scr = float(20 * np.log10(np.abs(tumor_peak) / (np.abs(clutter_mean) + 1e-10)))

    on_edge = (abs(abs(px) - grid_radius_mm) < 1e-6) or (abs(abs(py) - grid_radius_mm) < 1e-6)
    return le, scr, on_edge, px, py


def reconstruct_and_score_all_variants(scan_idx, s21, tumor_model, id_to_original_idx, verbose=False):
    """Computes the delay grid + time-domain signal ONCE, reuses across all
    4 variants (Raw DAS, Raw DMAS, DAS+CF, DMAS+CF), since physics doesn't
    depend on which beamformer is used.

    scan_idx here is a position in tumor_model (0..len(tumor_model)-1), NOT
    a position in s21 — those two are no longer the same after filtering.
    row['original_s21_idx'] is the corrected lookup into the real s21 array.
    """
    row = tumor_model.iloc[scan_idx]
    breast_radius_mm = float(row["breast_radius_mm"])
    fat_frac = float(row["fat_fraction"])
    fib_frac = float(row["fib_fraction"])
    v_tissue, _ = physics.compute_tissue_velocity(fat_frac, fib_frac)

    ant_rad_mm = float(row.get("ant_rad", 21.5)) * 10.0
    geom = physics.get_antenna_geometry(ant_rad_mm)

    s21_idx = int(row["original_s21_idx"])
    fd_scan = s21[s21_idx]
    fd_scan_mean_abs_before_cal = float(np.mean(np.abs(fd_scan)))

    # Empty-chamber calibration subtraction (Aurel's original
    # calibrate_iczt_complex approach) — subtract the reference scan of the
    # empty chamber, looked up via this scan's emp_ref_id, BEFORE ICZT.
    emp_ref_id = row.get("emp_ref_id", None)
    calibration_applied = False
    if emp_ref_id is not None and not pd.isna(emp_ref_id) and int(emp_ref_id) in id_to_original_idx:
        emp_idx = id_to_original_idx[int(emp_ref_id)]
        fd_scan = fd_scan - s21[emp_idx]
        calibration_applied = True

    diag = dict(
        original_s21_idx=s21_idx,
        old_buggy_idx=scan_idx,          # what indexing WOULD have used pre-fix
        idx_drift=s21_idx - scan_idx,     # 0 means the fix changed nothing for this scan
        calibration_applied=calibration_applied,
        phant_id=row["phant_id"],
        gt_x_mm=float(row["tumor_x_mm"]),
        gt_y_mm=float(row["tumor_y_mm"]),
        breast_radius_mm=breast_radius_mm,
        tum_in_fib=row.get("tum_in_fib", np.nan),
    )

    time_signal = sp.to_time_domain(fd_scan)

    if verbose:
        old_idx_fd_scan = s21[scan_idx]
        print(f"\n  [VERIFY scan {scan_idx}] phant_id={row['phant_id']}  "
              f"original_s21_idx={s21_idx}  old_buggy_idx={scan_idx}  emp_ref_id={emp_ref_id}")
        print(f"    mean|s21[old_buggy_idx]|        : {float(np.mean(np.abs(old_idx_fd_scan))):.6e}")
        print(f"    mean|s21[original_s21_idx]|     : {fd_scan_mean_abs_before_cal:.6e}  "
              f"(before calibration subtraction)")
        print(f"    mean|fd_scan| after calibration  : {float(np.mean(np.abs(fd_scan))):.6e}")
        print(f"    mean|time_signal| after ICZT     : {float(np.mean(np.abs(time_signal))):.6e}")
    time_axis = sp.get_time_axis(time_signal.shape[0])
    td_mag = np.abs(time_signal)

    grid_x_mm, grid_y_mm, axis_mm, grid_radius_mm = build_grid(breast_radius_mm)
    grid_x_m = grid_x_mm.ravel() / 1000.0
    grid_y_m = grid_y_mm.ravel() / 1000.0

    delay_grid = physics.two_medium_delay(
        geom["ant_x"], geom["ant_y"], geom["ant_x_b"], geom["ant_y_b"],
        grid_x_m, grid_y_m, breast_radius_mm / 1000.0, v_tissue,
    )
    delay_grid = delay_grid.reshape(-1, *grid_x_mm.shape)

    if verbose:
        # First-principles geometry check: at the exact center pixel, the
        # delay for any channel should be IDENTICAL to every other channel
        # (all antennas sit on the same ring radius, so distance-to-center
        # is constant regardless of angle) — this tests the antenna-ring
        # symmetry claim directly against delay_grid itself, independent of
        # any signal data. At a random OFF-center pixel, delays should differ
        # channel to channel, for contrast.
        center_iy = np.argmin(np.abs(axis_mm))
        center_ix = center_iy
        check_channels = [0, 18, 36, 54]
        center_delays = [delay_grid[ch, center_iy, center_ix] for ch in check_channels]
        off_iy = min(len(axis_mm) - 1, center_iy + 30)   # ~7.5mm off-center at 0.25mm grid
        off_ix = center_ix
        off_delays = [delay_grid[ch, off_iy, off_ix] for ch in check_channels]
        print(f"    [GEOMETRY CHECK] delay at center (channels {check_channels}): "
              f"{[f'{d*1e9:.4f}ns' for d in center_delays]}")
        print(f"    [GEOMETRY CHECK] delay off-center (same channels)      : "
              f"{[f'{d*1e9:.4f}ns' for d in off_delays]}")

    gt_x_mm = float(row["tumor_x_mm"])
    gt_y_mm = float(row["tumor_y_mm"])

    images = {
        "Raw DAS": bf.das_coherent(time_signal, time_axis, delay_grid),
        "Raw DMAS": bf.dmas(td_mag, time_axis, delay_grid),
    }
    das_image_raw, cf_map, das_cf_img = bf.das_coherent_cf(time_signal, time_axis, delay_grid)
    images["DAS+CF"] = das_cf_img
    dmas_image_raw = bf.dmas(td_mag, time_axis, delay_grid)
    dmas_cf_img, _ = bf.dmas_cf(time_signal, td_mag, time_axis, delay_grid)
    images["DMAS+CF"] = dmas_cf_img

    # Debiased CF variants — subtract out the geometry-only bias measured
    # from 10 pure-noise trials on this phantom's own delay_grid (see chat
    # discussion: the ring's rotational symmetry inflates CF near center
    # regardless of any real signal; this baseline isolates exactly how
    # much, per pixel, so it can be corrected out rather than just masked).
    n_ant = time_signal.shape[1]
    baseline_mean, baseline_std = get_cf_baseline_cached(
        row["phant_id"], delay_grid, time_axis, n_ant)

    cf_sub = debias_cf_subtraction(cf_map, baseline_mean)
    cf_ratio = debias_cf_ratio(cf_map, baseline_mean)
    cf_zscore = debias_cf_zscore(cf_map, baseline_mean, baseline_std)

    images["DAS+CF-debiased-sub"] = das_image_raw * cf_sub
    images["DAS+CF-debiased-ratio"] = das_image_raw * cf_ratio
    images["DAS+CF-debiased-zscore"] = das_image_raw * cf_zscore
    images["DMAS+CF-debiased-sub"] = dmas_image_raw * cf_sub
    images["DMAS+CF-debiased-ratio"] = dmas_image_raw * cf_ratio
    images["DMAS+CF-debiased-zscore"] = dmas_image_raw * cf_zscore

    if verbose:
        center_idx = np.argmin(np.abs(axis_mm))
        cf_at_center = cf_map[center_idx, center_idx]
        cf_elsewhere_mean = np.mean(np.delete(cf_map.ravel(),
                                     center_idx * cf_map.shape[1] + center_idx))
        print(f"    CF at center pixel: {cf_at_center:.4f}   "
              f"CF mean elsewhere: {cf_elsewhere_mean:.4f}")
        print(f"    [BASELINE] CF-noise-baseline at center: mean={baseline_mean[center_idx, center_idx]:.4f}  "
              f"std={baseline_std[center_idx, center_idx]:.4f}")
        print(f"    [DAS+CF] peak brightness value (img.max()): {images['DAS+CF'].max():.6e}")
        print(f"    [Raw DAS] peak brightness value (img.max()): {images['Raw DAS'].max():.6e}")

    CENTROID_TEST_VARIANTS = ("Raw DAS", "DAS+CF", "DAS+CF-debiased-zscore")

    results = {}
    for name, img in images.items():
        le, scr, on_edge, px, py = argmax_localize_and_score(
            img, axis_mm, grid_radius_mm, gt_x_mm, gt_y_mm)
        results[name] = dict(localization_error_mm=le, scr_db=scr, on_edge=on_edge)

        le_m, scr_m, on_edge_m, px_m, py_m = argmax_localize_and_score(
            img, axis_mm, grid_radius_mm, gt_x_mm, gt_y_mm,
            exclude_center_radius_mm=CENTER_EXCLUDE_RADIUS_MM)
        results[f"{name} (masked)"] = dict(localization_error_mm=le_m, scr_db=scr_m, on_edge=on_edge_m)

        if name in CENTROID_TEST_VARIANTS:
            le_c, scr_c, on_edge_c, px_c, py_c = weighted_centroid_localize_and_score(
                img, axis_mm, grid_radius_mm, gt_x_mm, gt_y_mm)
            results[f"{name} (centroid)"] = dict(
                localization_error_mm=le_c, scr_db=scr_c, on_edge=on_edge_c)

            le_b, scr_b, on_edge_b, px_b, py_b = largest_blob_localize_and_score(
                img, axis_mm, grid_radius_mm, gt_x_mm, gt_y_mm)
            results[f"{name} (largest-blob)"] = dict(
                localization_error_mm=le_b, scr_db=scr_b, on_edge=on_edge_b)

            if verbose:
                print(f"    [{name} centroid] peak=({px_c:.2f},{py_c:.2f})mm  LE={le_c:.2f}mm")
                print(f"    [{name} largest-blob] peak=({px_b:.2f},{py_b:.2f})mm  LE={le_b:.2f}mm")

        if verbose and name in ("Raw DAS", "DAS+CF"):
            print(f"    [{name}] peak=({px:.2f},{py:.2f})mm  "
                  f"gt=({gt_x_mm:.2f},{gt_y_mm:.2f})mm  LE={le:.2f}mm")
            print(f"    [{name} masked] peak=({px_m:.2f},{py_m:.2f})mm  LE={le_m:.2f}mm")

    # Candidate generation for the SVM verification stage — DAS+CF image only.
    # shell_center defaults to (0,0), same as the ring center, since real
    # adi_x/adi_y offsets aren't wired into this script (known pending item —
    # dist_to_shell_center and dist_to_ring_center will be numerically
    # identical here until that's added).
    birads = row.get("birads", np.nan)
    candidates = generate_candidates(
        images["DAS+CF"], cf_map, axis_mm, gt_x_mm, gt_y_mm,
        shell_center_mm=(0.0, 0.0), birads=birads, breast_radius_mm=breast_radius_mm,
        phant_id=row["phant_id"],
    )
    # k=10 recall check — reuses the same already-computed image/cf_map, so
    # this costs only extra peak-search + feature-extraction, no additional
    # beamforming (see chat discussion: does the true tumor show up more
    # often among more candidates, or is it missing from the image itself?)
    candidates_k10 = generate_candidates(
        images["DAS+CF"], cf_map, axis_mm, gt_x_mm, gt_y_mm,
        shell_center_mm=(0.0, 0.0), birads=birads, breast_radius_mm=breast_radius_mm,
        phant_id=row["phant_id"],
        k=10,
    )

    if verbose:
        print(f"    [ALL {len(candidates)} CANDIDATES] gt=({gt_x_mm:.2f},{gt_y_mm:.2f})mm")
        for i, c in enumerate(candidates):
            print(f"      #{i}: peak=({c['peak_x_mm']:.2f},{c['peak_y_mm']:.2f})mm  "
                  f"le_to_gt={c['le_to_gt_mm']:.2f}mm  label={c['label']}  "
                  f"cf_at_peak={c['cf_at_peak']:.3f}  dist_to_ring_center={c['dist_to_ring_center_mm']:.2f}mm")

    return results, diag, candidates, candidates_k10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-scans", type=int, default=None,
                         help="Limit to a random sample of N scans (smoke test). "
                              "Default: all valid scans WITHIN the selected phantoms.")
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for phantom selection AND scan sampling "
                              "(reproducibility).")
    parser.add_argument("--n-phantoms", type=int, default=30,
                         help="Number of phantoms to restrict to, via select_phantoms.py's "
                              "stratified (size x BI-RADS) selection — decided BEFORE any "
                              "LE/SCR is computed, not based on performance. "
                              "Set to 0 to use all available phantoms instead.")
    parser.add_argument("--one-per-phantom", action="store_true",
                         help="Pick exactly ONE scan per selected phantom (randomly, seeded) "
                              "instead of drawing --n-scans randomly from the pooled set of "
                              "all scans across all phantoms. Guarantees full phantom "
                              "coverage with no repeats — e.g. with --n-phantoms 30, always "
                              "gives exactly 30 scans from 30 DISTINCT phantoms. Without this "
                              "flag, --n-scans random draws can repeat some phantoms and miss "
                              "others entirely (see chat discussion).")
    parser.add_argument("--n-jobs", type=int, default=1,
                         help="Parallel worker processes (joblib). 1=serial (default), "
                              "4=four workers running simultaneously (~4x speedup, matches "
                              "Ursula's setup), -1=use all available CPU cores.")
    parser.add_argument("--save-candidates", type=str, default=None,
                         help="If given a path (e.g. svm_candidates.csv), saves the full "
                              "k=10 candidate dataset (all features, labels, phant_id) to "
                              "that CSV for training the SVM in train_svm.py. Not saved by "
                              "default.")
    args = parser.parse_args()

    print("Loading data...")
    d = load_all_data()
    s21, tumor_model = d["s21"], d["tumor_model"]
    id_to_original_idx = d["id_to_original_idx"]

    # Restrict to the stratified phantom subset (see select_phantoms.py —
    # selection based ONLY on phantom size/BI-RADS, never on performance).
    # reset_index is REQUIRED here: everything downstream uses positional
    # .iloc[scan_idx] into tumor_model, and filtering without resetting
    # would silently reintroduce the exact row-position bug fixed earlier
    # in this project (see chat discussion — data_loading.py's alignment fix).
    if args.n_phantoms > 0:
        phantom_table = build_phantom_table(tumor_model)
        selected = stratified_select(phantom_table, n=args.n_phantoms, seed=args.seed)
        selected_ids = selected["phant_id"].tolist()
        tumor_model = tumor_model[tumor_model["phant_id"].isin(selected_ids)].reset_index(drop=True)
        print(f"\nRestricted to {len(selected_ids)} phantoms (seed={args.seed}, "
              f"stratified by size x BI-RADS, decided before any LE/SCR result):")
        print(f"  {selected_ids}")
        print(f"  -> {len(tumor_model)} scans available within these phantoms")
    else:
        print(f"\nUsing all {tumor_model['phant_id'].nunique()} available phantoms "
              f"({len(tumor_model)} scans) — no phantom restriction.")

    # RANDOM sample, not the first N — tumor_model groups multiple
    # tumor-position scans per phantom shell, so "the first N" would
    # over-represent just 1-2 phantom geometries rather than giving a
    # genuinely diverse sample (see chat discussion — this is exactly what
    # happened with the original first-20 verify scans, all phant_id=A1F11).
    rng = np.random.default_rng(args.seed)

    if args.one_per_phantom:
        # Exactly one scan per phantom, guaranteeing full distinct-phantom
        # coverage — a pooled random draw (the branch below) does NOT
        # guarantee this: it could pick 3 scans from one phantom and 0 from
        # another, since it samples uniformly over SCANS, not over PHANTOMS.
        scan_indices = []
        for pid, group in tumor_model.groupby("phant_id"):
            scan_indices.append(int(rng.choice(group.index.values)))
        scan_indices = np.array(sorted(scan_indices))
        print(f"\n--one-per-phantom: selected exactly 1 scan from each of "
              f"{len(scan_indices)} distinct phantoms")
    elif args.n_scans is not None and args.n_scans < len(tumor_model):
        scan_indices = rng.choice(len(tumor_model), size=args.n_scans, replace=False)
    else:
        scan_indices = np.arange(len(tumor_model))
    n_scans = len(scan_indices)

    print(f"\nRunning baseline argmax test: Raw DAS / Raw DMAS / DAS+CF / DMAS+CF "
          f"on {n_scans} RANDOMLY SAMPLED scans (seed={args.seed}) "
          f"(grid={GRID_STEP_MM}mm, no bent-ray, no TVSVD, argmax localization)\n")

    base_names = ("Raw DAS", "Raw DMAS", "DAS+CF", "DMAS+CF",
                  "DAS+CF-debiased-sub", "DAS+CF-debiased-ratio", "DAS+CF-debiased-zscore",
                  "DMAS+CF-debiased-sub", "DMAS+CF-debiased-ratio", "DMAS+CF-debiased-zscore")
    variant_rows = {name: [] for name in base_names}
    for name in base_names:
        variant_rows[f"{name} (masked)"] = []
    for name in ("Raw DAS", "DAS+CF", "DAS+CF-debiased-zscore"):
        variant_rows[f"{name} (centroid)"] = []
        variant_rows[f"{name} (largest-blob)"] = []
    diag_rows = []
    all_candidates = []
    all_candidates_k10 = []
    failed = []
    t_start = time_module.time()

    def _process_one(i, idx):
        """Runs in a worker process — verbose is decided by POSITION in the
        sample (i < 3), not completion order, since parallel workers finish
        out of order. NOTE: the per-phantom CF-noise-baseline cache
        (_CF_BASELINE_CACHE) is per-PROCESS, not shared across workers — each
        worker recomputes it the first time it sees a given phantom rather
        than reusing another worker's copy. Doesn't affect correctness, just
        means the speedup isn't perfectly linear with --n-jobs."""
        try:
            scan_results, diag, candidates, candidates_k10 = reconstruct_and_score_all_variants(
                idx, s21, tumor_model, id_to_original_idx, verbose=(i < 3))
            return ("ok", idx, scan_results, diag, candidates, candidates_k10)
        except Exception as e:
            return ("fail", idx, str(e))

    print(f"Running with --n-jobs={args.n_jobs} "
          f"({'parallel' if args.n_jobs != 1 else 'serial'})...")
    outcomes = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(_process_one)(i, int(idx)) for i, idx in enumerate(scan_indices)
    )

    for outcome in outcomes:
        if outcome[0] == "ok":
            _, idx, scan_results, diag, candidates, candidates_k10 = outcome
            for name, metrics in scan_results.items():
                variant_rows[name].append(metrics)
            diag["scan_idx"] = idx
            diag_rows.append(diag)
            for c in candidates:
                c["scan_idx"] = idx
            all_candidates.extend(candidates)
            for c in candidates_k10:
                c["scan_idx"] = idx
            all_candidates_k10.extend(candidates_k10)
        else:
            _, idx, err = outcome
            failed.append((idx, err))

    print(f"  {len(outcomes) - len(failed)}/{n_scans} scans done "
          f"({time_module.time() - t_start:.0f}s elapsed)")

    if failed:
        print(f"\n{len(failed)} scans failed. First failure: {failed[0]}")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC: index alignment fix + calibration engagement")
    print("=" * 70)
    diag_df = pd.DataFrame(diag_rows)
    if len(diag_df) > 0:
        n_drifted = (diag_df["idx_drift"] != 0).sum()
        print(f"Scans where original_s21_idx differs from the old buggy index : "
              f"{n_drifted}/{len(diag_df)}")
        print(f"  mean |drift|  : {diag_df['idx_drift'].abs().mean():.1f}")
        print(f"  max |drift|   : {diag_df['idx_drift'].abs().max()}")
        n_calibrated = diag_df["calibration_applied"].sum()
        print(f"Scans where empty-chamber calibration was actually applied : "
              f"{n_calibrated}/{len(diag_df)}")
        if n_calibrated < len(diag_df):
            print("  *** some scans silently skipped calibration — emp_ref_id "
                  "missing/NaN or not found in id_to_original_idx")

    print("\n" + "=" * 70)
    print("BASELINE ARGMAX TEST — SUMMARY (Aurel's original physics + localization)")
    print("=" * 70)
    for name, rows in variant_rows.items():
        if not rows:
            print(f"{name}: no successful scans")
            continue
        df = pd.DataFrame(rows)
        mean_le = df["localization_error_mm"].mean()
        median_le = df["localization_error_mm"].median()
        mean_scr = df["scr_db"].mean()
        detection_rate = (df["localization_error_mm"] <= DETECTION_THRESHOLD_MM).mean()
        edge_fraction = df["on_edge"].mean()
        print(f"\n{name}  (n={len(df)})")
        print(f"  mean LE   : {mean_le:.2f} mm")
        print(f"  median LE : {median_le:.2f} mm")
        print(f"  mean SCR  : {mean_scr:.1f} dB")
        print(f"  detection @20mm : {detection_rate:.1%}")
        print(f"  on-edge fraction : {edge_fraction:.1%}")

    total_elapsed = time_module.time() - t_start
    print(f"\nTotal runtime: {total_elapsed:.0f}s")

    print("\n" + "=" * 70)
    print("SCR OUTLIER DIAGNOSTIC (DAS+CF, DMAS+CF)")
    print("=" * 70)
    diag_df_indexed = pd.DataFrame(diag_rows)
    for name in ("DAS+CF", "DAS+CF-debiased-sub", "DAS+CF-debiased-ratio", "DAS+CF-debiased-zscore",
                 "DMAS+CF", "DMAS+CF-debiased-sub", "DMAS+CF-debiased-ratio", "DMAS+CF-debiased-zscore"):
        rows = variant_rows.get(name, [])
        if not rows or len(rows) != len(diag_df_indexed):
            print(f"{name}: row count mismatch with diag — skipping outlier detail")
            continue
        scr_series = pd.Series([r["scr_db"] for r in rows])
        le_series = pd.Series([r["localization_error_mm"] for r in rows])
        print(f"\n{name} SCR distribution (dB):")
        print(f"  min={scr_series.min():.1f}  p25={scr_series.quantile(.25):.1f}  "
              f"median={scr_series.median():.1f}  p75={scr_series.quantile(.75):.1f}  "
              f"max={scr_series.max():.1f}")
        n_negative = (scr_series < 0).sum()
        print(f"  scans with negative SCR : {n_negative}/{len(scr_series)}")

        worst_n = min(5, len(scr_series))
        worst_idx = scr_series.nsmallest(worst_n).index
        print(f"  worst {worst_n} scans by SCR:")
        for i in worst_idx:
            d = diag_df_indexed.iloc[i]
            print(f"    scan_idx={d['scan_idx']}  phant_id={d['phant_id']}  "
                  f"gt=({d['gt_x_mm']:.1f},{d['gt_y_mm']:.1f})mm  "
                  f"breast_radius={d['breast_radius_mm']:.1f}mm  "
                  f"SCR={scr_series[i]:.1f}dB  LE={le_series[i]:.1f}mm")

        # Correlation check: does SCR get worse as the tumor sits farther
        # from the ring center (as a fraction of the phantom's own radius,
        # so different-sized phantoms are comparable)? Motivated by the
        # worst-5 pattern seen in the 20-scan run — all 87-96% of the way
        # to the phantom boundary.
        gt_dist_from_center = np.hypot(diag_df_indexed["gt_x_mm"], diag_df_indexed["gt_y_mm"])
        dist_ratio = gt_dist_from_center / diag_df_indexed["breast_radius_mm"]
        pearson_r = float(np.corrcoef(dist_ratio, scr_series)[0, 1])
        spearman_r = float(pd.Series(dist_ratio.values).corr(scr_series, method="spearman"))
        print(f"\n  Correlation: tumor distance-from-center ratio vs SCR")
        print(f"    Pearson r  : {pearson_r:.3f}")
        print(f"    Spearman r : {spearman_r:.3f}")
        print(f"    (negative r = tumors farther from center have worse SCR, "
              f"supporting the boundary-tumor theory)")

        # Binned breakdown for interpretability
        bins = pd.cut(dist_ratio, bins=[0, 0.5, 0.75, 0.9, 1.5],
                       labels=["0-50%", "50-75%", "75-90%", "90%+"])
        binned = pd.DataFrame({"dist_ratio_bin": bins, "scr_db": scr_series.values})
        print(f"    Mean SCR by distance-from-center bin (% of phantom radius):")
        for label, group in binned.groupby("dist_ratio_bin", observed=True):
            print(f"      {label}: mean SCR={group['scr_db'].mean():.1f}dB  n={len(group)}")

        # Is phantom SIZE itself an independent risk factor, separate from
        # the distance-ratio effect above? Motivated by A1F1/A1F3 (the
        # smallest phantom shell, 22.9mm radius) showing up in the worst-5
        # at n=50 despite the ratio already normalizing for size — checking
        # whether raw breast_radius_mm alone still correlates with SCR.
        radius_series = diag_df_indexed["breast_radius_mm"]
        radius_pearson_r = float(np.corrcoef(radius_series, scr_series)[0, 1])
        print(f"\n  Correlation: phantom breast_radius_mm ALONE vs SCR (independent of ratio)")
        print(f"    Pearson r  : {radius_pearson_r:.3f}")
        print(f"    (if this is also meaningfully negative, phantom size may be an "
              f"independent risk factor beyond what the distance ratio already captures)")

    print("\n" + "=" * 70)
    print("TUM_IN_FIB STRATIFICATION (fibroglandular-embedded vs. not)")
    print("=" * 70)
    print("Testing whether tumors inside the fibroglandular structure fare worse —")
    print("our two-medium model uses ONE blended tissue velocity for the whole")
    print("phantom interior, which doesn't account for WHERE the (locally denser,")
    print("slower) fibroglandular structure actually sits. A tumor embedded in it")
    print("would see a bigger local mismatch between assumed and true velocity.")
    for name in ("DAS+CF", "DAS+CF-debiased-zscore"):
        rows = variant_rows.get(name, [])
        if not rows or len(rows) != len(diag_df_indexed):
            print(f"\n{name}: row count mismatch — skipping")
            continue
        le_series = pd.Series([r["localization_error_mm"] for r in rows])
        scr_series = pd.Series([r["scr_db"] for r in rows])
        strat_df = pd.DataFrame({
            "tum_in_fib": diag_df_indexed["tum_in_fib"].values,
            "le_mm": le_series.values,
            "scr_db": scr_series.values,
        })
        print(f"\n{name}:")
        for value, group in strat_df.groupby("tum_in_fib", dropna=False):
            label = {0: "outside fibroglandular", 1: "inside fibroglandular"}.get(
                value, f"unknown/NaN ({value})")
            print(f"  {label}: n={len(group)}  mean LE={group['le_mm'].mean():.1f}mm  "
                  f"mean SCR={group['scr_db'].mean():.1f}dB")

    print("\n" + "=" * 70)
    print("SVM CANDIDATE DATASET SUMMARY (DAS+CF image, top-{} peaks/scan)".format(TOP_K_CANDIDATES))
    print("=" * 70)
    cand_df = pd.DataFrame(all_candidates)
    if len(cand_df) > 0:
        n_pos = int((cand_df["label"] == 1).sum())
        n_neg = int((cand_df["label"] == 0).sum())
        print(f"Total candidates : {len(cand_df)}  (from {n_scans} scans, "
              f"{len(cand_df) / n_scans:.1f} avg/scan)")
        print(f"Positive (label=1, real tumor)     : {n_pos}  ({n_pos / len(cand_df):.1%})")
        print(f"Negative (label=0, false positive) : {n_neg}  ({n_neg / len(cand_df):.1%})")
        print(f"Scans with >=1 positive candidate  : {cand_df.groupby('scan_idx')['label'].max().sum()}/{n_scans}")

        feature_cols = ["cf_at_peak", "cf_background_ratio", "area_fraction",
                         "dist_to_shell_center_mm", "dist_to_ring_center_mm",
                         "dist_to_ring_center_ratio", "local_scr_db", "hessian_eigenratio"]
        print("\nFeature ranges (min / mean / max), split by label:")
        for col in feature_cols:
            pos_vals = cand_df.loc[cand_df["label"] == 1, col]
            neg_vals = cand_df.loc[cand_df["label"] == 0, col]
            print(f"  {col}:")
            if len(pos_vals) > 0:
                print(f"    label=1 : {pos_vals.min():.3f} / {pos_vals.mean():.3f} / {pos_vals.max():.3f}")
            else:
                print(f"    label=1 : (no positive candidates)")
            print(f"    label=0 : {neg_vals.min():.3f} / {neg_vals.mean():.3f} / {neg_vals.max():.3f}")

        # --- Recall at k=10 vs k=5: is the ceiling on "tumor in candidate
        # list at all" fixable just by looking at more peaks? (see chat
        # discussion — caps how good ANY ranking approach can possibly do)
        cand_df_k10 = pd.DataFrame(all_candidates_k10)
        recall_k5 = cand_df.groupby("scan_idx")["label"].max().sum() / n_scans
        recall_k10 = (cand_df_k10.groupby("scan_idx")["label"].max().sum() / n_scans
                      if len(cand_df_k10) > 0 else float("nan"))
        print(f"\nCandidate recall (fraction of scans where the true tumor is "
              f"SOMEWHERE in the top-K list):")
        print(f"  k=5  : {recall_k5:.1%}")
        print(f"  k=10 : {recall_k10:.1%}")
        print(f"  (if k=10 isn't meaningfully higher than k=5, the tumor is often "
              f"missing from the image's local maxima entirely — a reconstruction-side "
              f"problem, not something more candidates can fix)")

        # --- AXIS ORIENTATION / ORIGIN CHECK (see chat discussion): if our
        # x/y axes were swapped or a sign were flipped relative to the
        # dataset's tum_x/tum_y convention, correctly-matched (label=1)
        # candidates would show a systematic, consistent directional bias
        # in their signed residuals (predicted - true) — NOT scatter
        # randomly around zero. Uses the k=10 pool for more label=1 examples.
        matched = cand_df_k10[cand_df_k10["label"] == 1]
        print(f"\n{'=' * 70}")
        print(f"AXIS ORIENTATION / ORIGIN CHECK (n={len(matched)} correctly-matched candidates)")
        print(f"{'=' * 70}")
        if len(matched) >= 3:
            rx_mean, rx_std = matched["residual_x_mm"].mean(), matched["residual_x_mm"].std()
            ry_mean, ry_std = matched["residual_y_mm"].mean(), matched["residual_y_mm"].std()
            print(f"  residual_x (predicted - true): mean={rx_mean:+.2f}mm  std={rx_std:.2f}mm")
            print(f"  residual_y (predicted - true): mean={ry_mean:+.2f}mm  std={ry_std:.2f}mm")
            # flag only if the mean bias is large relative to its own spread —
            # a small mean with wide scatter is normal noise, not a systematic bug
            x_suspicious = abs(rx_mean) > rx_std if rx_std > 0 else abs(rx_mean) > 1
            y_suspicious = abs(ry_mean) > ry_std if ry_std > 0 else abs(ry_mean) > 1
            if x_suspicious or y_suspicious:
                print(f"  *** SUSPICIOUS: mean residual exceeds its own std dev in "
                      f"{'x' if x_suspicious else ''}{' and ' if x_suspicious and y_suspicious else ''}"
                      f"{'y' if y_suspicious else ''} — worth investigating a systematic "
                      f"axis/sign issue further.")
            else:
                print(f"  No systematic directional bias detected — residuals scatter "
                      f"around zero within normal noise, consistent with CORRECT axis "
                      f"orientation and origin (a swap/sign-flip bug would show a large, "
                      f"consistent offset here instead).")
        else:
            print(f"  Too few correctly-matched candidates (n={len(matched)}) in this "
                  f"sample for a reliable check — rerun with more scans.")

        # --- THE DIRECT TEST: does re-ranking candidates by dist_ratio (our
        # strongest feature) actually improve LE, or just AUC on labels?
        # This is the real question the whole feature-AUC investigation has
        # been building toward — everything above measured a proxy metric,
        # this measures the actual thing we care about (see chat discussion).
        # Reuses the already-generated k=10 candidates — no new reconstructions.
        if len(cand_df_k10) > 0:
            argmax_les, dist_ratio_les, combined_les = [], [], []
            for scan_idx_g, group in cand_df_k10.groupby("scan_idx"):
                group = group.sort_index()   # restore original (brightest-first) order
                argmax_les.append(group.iloc[0]["le_to_gt_mm"])   # candidate #0 = plain argmax
                dist_ratio_les.append(group.loc[group["dist_to_ring_center_ratio"].idxmax(),
                                                 "le_to_gt_mm"])
                # corrected combined score: cf_at_peak flipped (see AUC section above —
                # low cf_at_peak was found to predict real tumor, not high)
                combined = (1 - group["cf_at_peak"]) * group["dist_to_ring_center_ratio"]
                combined_les.append(group.loc[combined.idxmax(), "le_to_gt_mm"])

            argmax_les = pd.Series(argmax_les)
            dist_ratio_les = pd.Series(dist_ratio_les)
            combined_les = pd.Series(combined_les)

            print(f"\n{'=' * 70}")
            print(f"DIRECT LE TEST: re-ranking vs plain argmax (k=10 candidate pool, "
                  f"n={len(argmax_les)} scans)")
            print(f"{'=' * 70}")
            for name_l, series_l in [("Plain argmax (candidate #0)", argmax_les),
                                      ("Re-ranked by dist_ratio (max wins)", dist_ratio_les),
                                      ("Re-ranked by (1-cf)*dist_ratio (max wins)", combined_les)]:
                print(f"  {name_l:<42}: mean LE={series_l.mean():.2f}mm  "
                      f"median LE={series_l.median():.2f}mm  "
                      f"detection@20mm={(series_l <= DETECTION_THRESHOLD_MM).mean():.1%}")
            print(f"  (if re-ranked LE is meaningfully lower than plain argmax, "
                  f"candidate ranking genuinely helps localization — not just labels)")

        # --- Univariate separability (AUC-ROC via rank-sum, no sklearn
        # dependency) for each feature, plus one hand-combined score
        # (cf_at_peak x dist_to_ring_center_ratio) to directly test whether
        # combining features beats either alone (see chat discussion — the
        # actual test of "is candidate ranking worth pursuing").
        def _auc_rank_sum(values, labels):
            n_pos = int((labels == 1).sum())
            n_neg = int((labels == 0).sum())
            if n_pos == 0 or n_neg == 0:
                return float("nan")
            ranks = pd.Series(values).rank().values
            rank_sum_pos = ranks[labels.values == 1].sum()
            return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

        def _auc_and_maybe_flip(values, labels):
            """If a feature's own AUC is below 0.5, use (1-value) instead so
            combined scores never get built by multiplying in a wrong-
            direction feature (see chat discussion — this was a real bug:
            multiplying by cf_at_peak/hessian_eigenratio directly, both of
            which measured wrong-direction, silently punished good
            candidates). Adaptive rather than hardcoded, since direction
            may hold, reverse, or be sample-size-dependent."""
            auc = _auc_rank_sum(values, labels)
            if auc < 0.5:
                return 1.0 - values, 1.0 - auc, True
            return values, auc, False

        print(f"\nUnivariate separability (AUC — 0.5=random, 1.0=perfect, "
              f"<0.5 means the feature points the WRONG direction):")
        labels_series = cand_df["label"]
        for col in feature_cols:
            auc = _auc_rank_sum(cand_df[col], labels_series)
            flag = "  <-- weak (near 0.5)" if 0.4 < auc < 0.6 else (
                "  <-- WRONG DIRECTION" if auc < 0.4 else "")
            print(f"  {col:<28}: AUC={auc:.3f}{flag}")

        cf_fixed, cf_auc_alone, cf_flipped = _auc_and_maybe_flip(
            cand_df["cf_at_peak"], labels_series)
        dist_ratio_fixed_a, dist_ratio_auc_alone_a, dist_flipped_a = _auc_and_maybe_flip(
            cand_df["dist_to_ring_center_ratio"], labels_series)
        combined_score = cf_fixed * dist_ratio_fixed_a
        combined_auc = _auc_rank_sum(combined_score, labels_series)
        best_single_auc = max(_auc_rank_sum(cand_df[c], labels_series) for c in feature_cols)
        print(f"\n  {'cf_at_peak * dist_ratio (combined, direction-corrected)':<28}: "
              f"AUC={combined_auc:.3f}  (best single feature alone: {best_single_auc:.3f})")
        print(f"    cf_at_peak alone: AUC={cf_auc_alone:.3f}"
              f"{' (flipped: used 1-value)' if cf_flipped else ''}")
        print(f"  (if combined AUC clearly beats best single feature, combining "
              f"features has real promise — if not, ranking may just reproduce argmax)")

        # Direct test of whether Hessian adds value ON TOP OF our current best
        # feature (dist_to_ring_center_ratio) — the actual question this
        # feature was added to answer.
        #
        # FIXED (was a sign-error bug): originally multiplied dist_ratio x
        # hessian_eigenratio directly, silently assuming "higher hessian is
        # better" — the same direction the n=20 run showed was WRONG
        # (AUC=0.395, real tumors scored LESS round than false positives).
        # Multiplying by a wrong-direction feature punishes good candidates
        # instead of rewarding them, which likely explains why the combined
        # score underperformed dist_ratio alone. Uses the same adaptive
        # _auc_and_maybe_flip helper defined above.
        dist_ratio_fixed, dist_ratio_auc_alone, dist_flipped = _auc_and_maybe_flip(
            cand_df["dist_to_ring_center_ratio"], labels_series)
        hessian_fixed, hessian_auc_alone, hessian_flipped = _auc_and_maybe_flip(
            cand_df["hessian_eigenratio"], labels_series)

        combined_ratio_hessian = dist_ratio_fixed * hessian_fixed
        combined_ratio_hessian_auc = _auc_rank_sum(combined_ratio_hessian, labels_series)
        print(f"\n  {'dist_ratio * hessian (combined, direction-corrected)':<28}: "
              f"AUC={combined_ratio_hessian_auc:.3f}")
        print(f"    dist_ratio alone: AUC={dist_ratio_auc_alone:.3f}"
              f"{' (flipped: used 1-value)' if dist_flipped else ''}")
        print(f"    hessian alone   : AUC={hessian_auc_alone:.3f}"
              f"{' (flipped: used 1-value)' if hessian_flipped else ''}")
        print(f"  (does adding shape information beat geometry alone, once both "
              f"features are pointed the correct direction? this is the "
              f"direct test of whether Hessian was worth adding)")
    else:
        print("No candidates generated.")

    # --- Save k=10 candidate dataset for train_svm.py, if requested ---
    if args.save_candidates:
        cand_df_k10_save = pd.DataFrame(all_candidates_k10)
        if len(cand_df_k10_save) > 0:
            cand_df_k10_save.to_csv(args.save_candidates, index=False)
            print(f"\nSaved {len(cand_df_k10_save)} candidates "
                  f"({cand_df_k10_save['scan_idx'].nunique()} scans, "
                  f"{cand_df_k10_save['phant_id'].nunique()} phantoms) to {args.save_candidates}")
        else:
            print(f"\nNo candidates to save — skipping --save-candidates={args.save_candidates}")


if __name__ == "__main__":
    main()