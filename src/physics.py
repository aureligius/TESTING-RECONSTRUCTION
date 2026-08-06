"""
src/physics.py

Antenna geometry + two delay models:
  - two_medium_delay: Aurel's air + adaptive-tissue-velocity model (line-circle
    intersection), used as the ablation baseline.
  - bent_ray_3layer_delay: Ursula's air -> skin -> interior Fermat-principle
    solver, used as the proposed model.

Both take the same (antenna_xy, pixel_grid) inputs and return a delay grid in
seconds, so pipeline.py can swap between them with one flag.
"""

import numpy as np

C_LIGHT = 3e8
EPSILON_AIR = 1.0006
V_AIR = C_LIGHT / np.sqrt(EPSILON_AIR)   # ~299.9 mm/ns

N_ANT = 72
SEPARATION_DEG = 60.0


# ============================================================================
# Antenna geometry
# ============================================================================
def get_corrected_ant_radius_m(raw_rad_mm):
    """Rodriguez-Herrera (2016) antenna radius correction. Input mm, output metres."""
    return (0.97 * (raw_rad_mm - 0.106) + 0.148) / 1000.0


def get_antenna_geometry(ant_rad_mm, n_ant=N_ANT, separation_deg=SEPARATION_DEG,
                          apply_correction=True):
    """
    Per-scan antenna positions (both ports) from a raw ant_rad metadata value
    (mm, already converted from the metadata's native cm). Applies the
    Rodriguez-Herrera correction by default.

    Returns dict: ant_x, ant_y, ant_x_b, ant_y_b (arrays, length n_ant, metres),
    tx_idx, rx_idx (channel-mapping indices).
    """
    ant_rad_m = get_corrected_ant_radius_m(ant_rad_mm) if apply_correction else ant_rad_mm / 1000.0

    angles = np.linspace(0, -2 * np.pi, n_ant, endpoint=False)
    ant_x = ant_rad_m * np.cos(angles)
    ant_y = ant_rad_m * np.sin(angles)

    offset = np.deg2rad(separation_deg)
    ant_x_b = ant_rad_m * np.cos(angles + offset)
    ant_y_b = ant_rad_m * np.sin(angles + offset)

    sep_steps = int(round(separation_deg / 360.0 * n_ant))
    tx_idx = np.arange(n_ant)
    rx_idx = (np.arange(n_ant) + sep_steps) % n_ant

    return dict(ant_x=ant_x, ant_y=ant_y, ant_x_b=ant_x_b, ant_y_b=ant_y_b,
                tx_idx=tx_idx, rx_idx=rx_idx, ant_rad_m=ant_rad_m)


# ============================================================================
# Tissue permittivity / velocity
# ============================================================================
def compute_tissue_velocity(fat_fraction, fib_fraction, eps_fat=7.0, eps_fib=45.0):
    eps_tissue = fat_fraction * eps_fat + fib_fraction * eps_fib
    v_tissue = C_LIGHT / np.sqrt(eps_tissue)
    return v_tissue, eps_tissue


# ============================================================================
# Two-medium model (air + adaptive tissue velocity), line-circle intersection.
# Used as the ablation baseline against the 3-layer bent-ray model.
# ============================================================================
def _leg_time_two_medium(p0x, p0y, grid_x, grid_y, shell_radius_m, v_air, v_tissue,
                          shell_center=(0.0, 0.0)):
    cx, cy = shell_center
    p0x_s, p0y_s = p0x - cx, p0y - cy
    gx_s, gy_s = grid_x - cx, grid_y - cy

    dx = gx_s - p0x_s
    dy = gy_s - p0y_s
    seg_len = np.sqrt(dx ** 2 + dy ** 2)

    a = dx ** 2 + dy ** 2
    b = 2.0 * (p0x_s * dx + p0y_s * dy)
    c = p0x_s ** 2 + p0y_s ** 2 - shell_radius_m ** 2

    disc = b ** 2 - 4.0 * a * c
    valid = disc >= 0

    a_safe = np.where(a == 0, 1e-30, a)
    sqrt_disc = np.zeros_like(dx)
    sqrt_disc[valid] = np.sqrt(disc[valid])

    t1 = (-b - sqrt_disc) / (2.0 * a_safe)
    t2 = (-b + sqrt_disc) / (2.0 * a_safe)
    t_lo = np.clip(np.minimum(t1, t2), 0.0, 1.0)
    t_hi = np.clip(np.maximum(t1, t2), 0.0, 1.0)

    tissue_frac = np.where(valid, np.maximum(t_hi - t_lo, 0.0), 0.0)
    dist_tissue = tissue_frac * seg_len
    dist_air = seg_len - dist_tissue

    return dist_air / v_air + dist_tissue / v_tissue


def two_medium_delay(ant_x, ant_y, ant_x_b, ant_y_b, grid_x, grid_y,
                      shell_radius_m, v_tissue, shell_center=(0.0, 0.0),
                      v_air=V_AIR):
    """
    Bistatic delay grid (n_ant, n_pix) for every antenna pair, air outside the
    phantom shell + single adaptive tissue velocity inside. grid_x/grid_y are
    flat pixel-coordinate arrays in metres.
    """
    n_ant = len(ant_x)
    n_pix = grid_x.shape[-1] if grid_x.ndim > 1 else len(grid_x)
    delay = np.zeros((n_ant, n_pix))
    for i in range(n_ant):
        t_a = _leg_time_two_medium(ant_x[i], ant_y[i], grid_x, grid_y,
                                    shell_radius_m, v_air, v_tissue, shell_center)
        t_b = _leg_time_two_medium(ant_x_b[i], ant_y_b[i], grid_x, grid_y,
                                    shell_radius_m, v_air, v_tissue, shell_center)
        delay[i] = t_a + t_b
    return delay


# ============================================================================
# 3-layer bent-ray model (air -> skin -> interior), Fermat-principle /
# golden-section search. Used as the proposed model.
# ============================================================================
def _find_refraction_point(a_xy, t_xy, r_circle, v_out, v_in, n_iter=40):
    a_xy = np.asarray(a_xy, dtype=float)
    t_xy = np.asarray(t_xy, dtype=float)
    ang_a = np.arctan2(a_xy[..., 1], a_xy[..., 0])
    ang_t = np.arctan2(t_xy[..., 1], t_xy[..., 0])

    lo = np.minimum(ang_a, ang_t) - 0.75
    hi = np.maximum(ang_a, ang_t) + 0.75

    gr = (np.sqrt(5.0) - 1.0) / 2.0

    def travel_time(phi):
        bx = r_circle * np.cos(phi)
        by = r_circle * np.sin(phi)
        d1 = np.sqrt((bx - a_xy[..., 0]) ** 2 + (by - a_xy[..., 1]) ** 2)
        d2 = np.sqrt((bx - t_xy[..., 0]) ** 2 + (by - t_xy[..., 1]) ** 2)
        return d1 / v_out + d2 / v_in

    a, b = lo, hi
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = travel_time(c), travel_time(d)

    for _ in range(n_iter):
        go_left = fc < fd
        b = np.where(go_left, d, b)
        a = np.where(go_left, a, c)
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc, fd = travel_time(c), travel_time(d)

    phi_opt = 0.5 * (a + b)
    return np.stack([r_circle * np.cos(phi_opt), r_circle * np.sin(phi_opt)], axis=-1)


def bent_ray_3layer_delay(ant_x, ant_y, ant_x_b, ant_y_b, grid_x_flat, grid_y_flat,
                           breast_radius_m, skin_thickness_m,
                           v_air, v_skin, v_interior, fixed_point_iters=3):
    """
    3-layer bistatic delay grid (n_ant, n_pix): air -> skin -> interior on
    each leg (antenna -> pixel), solved via a fixed-point iteration over two
    Fermat-principle refraction solves per leg.

    Vectorized across ALL antennas simultaneously (no Python loop over
    n_ant) — _find_refraction_point already broadcasts correctly over
    arbitrary leading dimensions, so batching every antenna into one call
    removes ~72x the Python-interpreter overhead of the golden-section
    search versus calling it once per antenna. This was the main cost:
    the per-iteration math is cheap, the loop overhead was not.

    Note: does NOT include the GT-tumor 4th layer used in Ursula's calibration
    sweeps (use_gt_tumor_layer) — that mode is calibration-only and must not
    be used for inference. This function is inference-safe.
    """
    n_ant = len(ant_x)
    n_pix = len(grid_x_flat)
    r_outer = breast_radius_m
    r_inner = max(breast_radius_m - skin_thickness_m, 1e-4)

    # pixel positions, broadcastable across the antenna axis: (1, n_pix, 2)
    p = np.stack([grid_x_flat, grid_y_flat], axis=-1)[None, :, :]
    p_b = np.broadcast_to(p, (n_ant, n_pix, 2))

    def leg_delay(ax, ay):
        a = np.stack([ax, ay], axis=-1)[:, None, :]              # (n_ant, 1, 2)
        a_b = np.broadcast_to(a, (n_ant, n_pix, 2))

        b1 = _find_refraction_point(a_b, p_b, r_outer, v_air, v_interior)
        for _ in range(fixed_point_iters):
            b2 = _find_refraction_point(b1, p_b, r_inner, v_skin, v_interior)
            b1 = _find_refraction_point(a_b, b2, r_outer, v_air, v_skin)

        d_air = np.linalg.norm(b1 - a_b, axis=-1)
        d_skin = np.linalg.norm(b2 - b1, axis=-1)
        d_interior = np.linalg.norm(p_b - b2, axis=-1)
        return d_air / v_air + d_skin / v_skin + d_interior / v_interior

    delay_tx = leg_delay(ant_x, ant_y)      # (n_ant, n_pix)
    delay_rx = leg_delay(ant_x_b, ant_y_b)  # (n_ant, n_pix)
    return delay_tx + delay_rx