"""
src/pipeline.py

reconstruct_scan() — self-contained, per-scan reconstruction. Everything
(geometry, delay grid, images) is local to the call, never a leaked global,
matching both Aurel's and Ursula's original per-scan design.

Toggle flags select which of the 8 ablation-matrix variants to run:
    beamformer:   'das' | 'dmas'
    use_bent_ray: False -> two_medium_delay (Aurel's baseline)
                  True  -> bent_ray_3layer_delay (Ursula's proposed model)
    use_cf:       False -> plain DAS/DMAS
                  True  -> CF-weighted

Calibrated 3-layer parameters (skin_thickness_m, eps_skin, eps_tumor) default
to literature values here; re-run Ursula's Phase 5.6-5.8 sensitivity sweeps
against this module if you want data-driven calibrated values instead
(pass them in via bent_ray_params).
"""

from time import perf_counter

import numpy as np

from . import physics
from . import signal_processing as sp
from . import beamforming as bf
from . import blob_detection as bd
from . import metrics as mx

DEFAULT_GRID_MARGIN_FACTOR = 1.5
DEFAULT_GRID_STEP_MM = 1.0

DEFAULT_BENT_RAY_PARAMS = dict(
    skin_thickness_mm=2.5,
    eps_skin=45.0,
)

# Delay grids depend only on phantom-level geometry (breast radius, antenna
# radius, tissue velocity, bent-ray params) — NOT on the per-scan signal.
# Since phantoms average ~15 scans each in this dataset, caching by phantom
# identity avoids recomputing an identical delay grid ~15x over.
_DELAY_CACHE = {}


def _delay_cache_key(phant_id, use_bent_ray, ant_rad_mm, breast_radius_mm,
                      v_tissue, bent_ray_params, margin_factor, grid_step_mm,
                      shell_center):
    if use_bent_ray:
        extra = tuple(sorted((bent_ray_params or {}).items()))
    else:
        extra = shell_center
    return (phant_id, use_bent_ray, round(ant_rad_mm, 3), round(breast_radius_mm, 3),
            round(v_tissue, 3), extra, margin_factor, grid_step_mm)


def build_grid(breast_radius_mm, margin_factor=DEFAULT_GRID_MARGIN_FACTOR,
                grid_step_mm=DEFAULT_GRID_STEP_MM):
    grid_radius_mm = breast_radius_mm * margin_factor
    axis_mm = np.arange(-grid_radius_mm, grid_radius_mm + grid_step_mm, grid_step_mm)
    grid_x_mm, grid_y_mm = np.meshgrid(axis_mm, axis_mm)
    return grid_x_mm, grid_y_mm, axis_mm, grid_radius_mm


def reconstruct_scan(scan_idx, s21, tumor_model,
                      beamformer="das", use_bent_ray=True, use_cf=True,
                      use_tvsvd=False,
                      bent_ray_params=None, shell_center=(0.0, 0.0),
                      margin_factor=DEFAULT_GRID_MARGIN_FACTOR,
                      grid_step_mm=DEFAULT_GRID_STEP_MM,
                      return_diagnostics=False):
    """
    Fully self-contained per-scan reconstruction.

    Parameters
    ----------
    scan_idx : int, row index into tumor_model / first axis of s21
    s21 : complex128 array (n_scans, n_freq, n_ant)
    tumor_model : DataFrame from data_loading.build_tumor_model()
    beamformer : 'das' | 'dmas'
    use_bent_ray : bool — 3-layer bent-ray delay model vs 2-medium baseline
    use_cf : bool — coherence-factor weighting on/off
    use_tvsvd : bool — hybrid TVSVD clutter suppression on/off. DEFAULT
                FALSE to match the "Raw+CF" baseline that produced 10.61mm
                on A16F14 in the original single-scan validation — earlier
                versions of this function applied TVSVD unconditionally,
                silently turning every "Raw" ablation-matrix label into a
                TVSVD-preprocessed run. Set True to test SVD-90%-equivalent
                variants explicitly, as their own row in the matrix.
    bent_ray_params : dict with skin_thickness_mm, eps_skin — overrides
                       DEFAULT_BENT_RAY_PARAMS if given
    shell_center : (x_mm, y_mm) offset of the phantom shell from chamber
                   origin — pass real adi_x/adi_y if available, else (0,0)

    Returns a dict with images, peak location, GT, and all metrics.
    """
    t_start = perf_counter()

    row = tumor_model.iloc[scan_idx]
    breast_radius_mm = float(row["breast_radius_mm"])
    fat_frac = float(row["fat_fraction"])
    fib_frac = float(row["fib_fraction"])
    v_tissue, eps_tissue = physics.compute_tissue_velocity(fat_frac, fib_frac)

    # ant_rad metadata field is confirmed in centimetres (see project notes /
    # metadata_info.md) — always convert to mm, no ambiguity check needed.
    ant_rad_cm = float(row.get("ant_rad", 21.5))
    ant_rad_mm = ant_rad_cm * 10.0
    geom = physics.get_antenna_geometry(ant_rad_mm)

    # ---- signal transform ----
    fd_scan = s21[scan_idx]  # (n_freq, n_ant)
    time_signal = sp.to_time_domain(fd_scan)
    time_axis = sp.get_time_axis(time_signal.shape[0])

    # ---- clutter suppression (optional — off by default, see use_tvsvd docstring) ----
    if use_tvsvd:
        time_signal_filtered, n_removed = sp.apply_hybrid_tvsvd(time_signal)
    else:
        time_signal_filtered, n_removed = time_signal, 0

    # ---- imaging grid (metres, for physics.py; mm for display/metrics) ----
    grid_x_mm, grid_y_mm, axis_mm, grid_radius_mm = build_grid(
        breast_radius_mm, margin_factor, grid_step_mm)
    grid_x_m, grid_y_m = grid_x_mm.ravel() / 1000.0, grid_y_mm.ravel() / 1000.0
    shell_center_m = (shell_center[0] / 1000.0, shell_center[1] / 1000.0)

    # ---- delay grid: two-medium baseline or 3-layer bent-ray (cached per phantom) ----
    cache_key = _delay_cache_key(
        row["phant_id"], use_bent_ray, ant_rad_mm, breast_radius_mm, v_tissue,
        bent_ray_params, margin_factor, grid_step_mm, shell_center,
    )
    if cache_key in _DELAY_CACHE:
        delay_grid = _DELAY_CACHE[cache_key]
    else:
        if use_bent_ray:
            params = {**DEFAULT_BENT_RAY_PARAMS, **(bent_ray_params or {})}
            v_skin = physics.C_LIGHT / np.sqrt(params["eps_skin"])
            delay_grid = physics.bent_ray_3layer_delay(
                geom["ant_x"], geom["ant_y"], geom["ant_x_b"], geom["ant_y_b"],
                grid_x_m, grid_y_m,
                breast_radius_mm / 1000.0, params["skin_thickness_mm"] / 1000.0,
                physics.V_AIR, v_skin, v_tissue,
            )
        else:
            delay_grid = physics.two_medium_delay(
                geom["ant_x"], geom["ant_y"], geom["ant_x_b"], geom["ant_y_b"],
                grid_x_m, grid_y_m,
                breast_radius_mm / 1000.0, v_tissue, shell_center=shell_center_m,
            )
        delay_grid = delay_grid.reshape(-1, *grid_x_mm.shape)
        _DELAY_CACHE[cache_key] = delay_grid

    # ---- beamforming ----
    if beamformer == "das":
        if use_cf:
            _, cf_map, img = bf.das_coherent_cf(time_signal_filtered, time_axis, delay_grid)
        else:
            img = bf.das_coherent(time_signal_filtered, time_axis, delay_grid)
            cf_map = None
    elif beamformer == "dmas":
        td_mag = np.abs(time_signal_filtered)
        if use_cf:
            img, cf_map = bf.dmas_cf(time_signal_filtered, td_mag, time_axis, delay_grid)
        else:
            img = bf.dmas(td_mag, time_axis, delay_grid)
            cf_map = None
    else:
        raise ValueError(f"Unknown beamformer: {beamformer!r} (expected 'das' or 'dmas')")

    # ---- blob extraction + localization ----
    blob = bd.extract_blob_candidate(img, axis_mm, axis_mm)

    gt_x_mm = float(row["tumor_x_mm"])
    gt_y_mm = float(row["tumor_y_mm"])
    gt_r_mm = float(row["tumor_radius_mm"])

    computed = mx.compute_all_metrics(
        img, blob["tumor_mask"], axis_mm, axis_mm,
        blob["peak_x"], blob["peak_y"], gt_x_mm, gt_y_mm, gt_r_mm,
    )

    cf_at_peak = None
    if cf_map is not None:
        peak_iy = np.argmin(np.abs(axis_mm - blob["peak_y"]))
        peak_ix = np.argmin(np.abs(axis_mm - blob["peak_x"]))
        cf_at_peak = float(cf_map[peak_iy, peak_ix])

    runtime_sec = perf_counter() - t_start

    result = dict(
        scan_idx=scan_idx,
        phant_id=row["phant_id"],
        birads=row.get("birads", np.nan),
        beamformer=beamformer, use_bent_ray=use_bent_ray, use_cf=use_cf,
        breast_radius_mm=breast_radius_mm, grid_radius_mm=grid_radius_mm,
        tvsvd_removed=n_removed,
        peak_x_mm=blob["peak_x"], peak_y_mm=blob["peak_y"],
        gt_x_mm=gt_x_mm, gt_y_mm=gt_y_mm, gt_r_mm=gt_r_mm,
        blob_area_px=blob["blob_area_px"], blob_compactness=blob["blob_compactness"],
        cf_at_peak=cf_at_peak,
        runtime_sec=runtime_sec,
        **computed,
    )

    if return_diagnostics:
        result["diagnostics"] = dict(
            image=img, cf_map=cf_map, tumor_mask=blob["tumor_mask"],
            axis_mm=axis_mm, time_signal=time_signal,
            time_signal_filtered=time_signal_filtered, delay_grid=delay_grid,
        )

    return result