"""
src/beamforming.py

DAS / DMAS beamforming + Coherence Factor (CF) weighting, with Aurel's
antenna-index Tukey windowing applied by default. Operates on an
already-computed delay grid (from physics.py, either two_medium_delay or
bent_ray_3layer_delay — this module doesn't care which) via interpolated
(not nearest-neighbor) delay lookup.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal.windows import tukey

ANTENNA_WINDOW_ALPHA = 0.2
GATE_NS = 0.5   # zero out samples before this — pre-tissue direct antenna
                 # coupling/crosstalk can't physically be a tissue reflection


def _gate_early_samples(time_signal, time_axis, gate_ns=GATE_NS):
    """Zero out time samples earlier than gate_ns — restores the pre-tissue
    time gate from the original recon_utils.py, which was dropped when
    beamforming.py was rewritten for the merge. Without this, direct
    antenna-to-antenna crosstalk (which arrives before any physically
    possible tissue reflection) can dominate the DAS/DMAS sum."""
    dt = time_axis[1] - time_axis[0]
    gate_idx = int((gate_ns * 1e-9) / dt)
    gated = time_signal.copy()
    gated[:gate_idx, :] = 0.0
    return gated


def get_antenna_window(n_ant, alpha=ANTENNA_WINDOW_ALPHA, enabled=True):
    """Tukey window across antenna index — reduces grating-lobe sidelobes
    from the sparse 72-position circular array. enabled=False -> uniform
    weights, for an explicit before/after comparison."""
    return tukey(n_ant, alpha=alpha) if enabled else np.ones(n_ant)


def _interp_lookup(time_signal_1d, time_axis, delay_seconds):
    """Complex-safe interpolation: signal at arbitrary delay times, zero-filled
    outside the recorded window. delay_seconds is the delay grid for a SINGLE
    antenna (antenna axis already indexed out by the caller), shape (ny, nx)
    or any spatial shape — reshape target must be its full shape, not
    shape[1:] (that was the bug: shape[1:] assumed a leftover antenna axis
    that the caller had already stripped)."""
    interp_re = interp1d(time_axis, time_signal_1d.real, bounds_error=False, fill_value=0.0)
    interp_im = interp1d(time_axis, time_signal_1d.imag, bounds_error=False, fill_value=0.0)
    flat = delay_seconds.ravel()
    return (interp_re(flat) + 1j * interp_im(flat)).reshape(delay_seconds.shape)


def das_coherent(time_signal, time_axis, delay_grid, window=None, gate_ns=GATE_NS):
    """
    Coherent DAS. time_signal: (n_time, n_ant) complex. delay_grid: (n_ant,
    n_pix) or (n_ant, ny, nx) seconds. Returns magnitude image, same spatial
    shape as delay_grid[0].
    """
    time_signal = _gate_early_samples(time_signal, time_axis, gate_ns)
    n_ant = time_signal.shape[1]
    if window is None:
        window = get_antenna_window(n_ant)

    accum = np.zeros(delay_grid.shape[1:], dtype=np.complex128)
    for i in range(n_ant):
        vals = _interp_lookup(time_signal[:, i], time_axis, delay_grid[i])
        accum += window[i] * vals
    return np.abs(accum)


def das_coherent_cf(time_signal, time_axis, delay_grid, window=None, cf_power=1.0,
                     gate_ns=GATE_NS):
    """
    Coherent DAS with Coherence Factor weighting.
    CF = |sum_i w_i*s_i|^2 / [(sum_i w_i^2) * sum_i |s_i|^2], bounded [0,1]
    by Cauchy-Schwarz. Near 1 = antennas agree in phase (real reflector);
    near 0 = random/no agreement (speckle/clutter).

    Returns (das_image, cf_map, cf_weighted_image) where
    cf_weighted_image = das_image * cf_map**cf_power.
    """
    time_signal = _gate_early_samples(time_signal, time_axis, gate_ns)
    n_ant = time_signal.shape[1]
    if window is None:
        window = get_antenna_window(n_ant)
    sum_w_sq = np.sum(window ** 2)

    sum_complex = np.zeros(delay_grid.shape[1:], dtype=np.complex128)
    sum_sq = np.zeros(delay_grid.shape[1:])

    for i in range(n_ant):
        vals = _interp_lookup(time_signal[:, i], time_axis, delay_grid[i])
        sum_complex += window[i] * vals
        sum_sq += np.abs(vals) ** 2

    das_image = np.abs(sum_complex)
    denom = sum_w_sq * sum_sq
    cf_map = np.where(denom > 1e-30, (das_image ** 2) / denom, 0.0)
    cf_map = np.clip(cf_map, 0.0, 1.0)

    cf_weighted_image = das_image * (cf_map ** cf_power)
    return das_image, cf_map, cf_weighted_image


def dmas(time_signal_magnitude, time_axis, delay_grid, subsample=3, window=None,
         gate_ns=GATE_NS):
    """
    DMAS on magnitude-only signal. Subsamples every `subsample`-th antenna,
    forms every unique pair among the subsample, sign-preserving product,
    sums. window applied per-antenna before pairing (propagates through the
    product naturally).
    """
    time_signal_magnitude = _gate_early_samples(time_signal_magnitude, time_axis, gate_ns)
    n_ant_total = time_signal_magnitude.shape[1]
    if window is None:
        window = get_antenna_window(n_ant_total)

    ant_indices = list(range(0, n_ant_total, subsample))
    lookup = {}
    for i in ant_indices:
        vals = _interp_lookup(time_signal_magnitude[:, i], time_axis, delay_grid[i])
        lookup[i] = window[i] * vals

    image = np.zeros(delay_grid.shape[1:])
    for idx_i, i in enumerate(ant_indices):
        s_i = lookup[i]
        for j in ant_indices[idx_i + 1:]:
            s_j = lookup[j]
            product = np.sign((s_i * s_j).real) * np.sqrt(np.abs(s_i * s_j))
            image += product
    return image


def dmas_cf(time_signal_complex, time_signal_magnitude, time_axis, delay_grid,
            subsample=3, window=None, cf_power=1.0):
    """DMAS image weighted by the CF map computed from the full-array
    coherent (complex) signal — reuses das_coherent_cf's cf_map rather than
    recomputing CF from the DMAS-subsampled channels."""
    _, cf_map, _ = das_coherent_cf(time_signal_complex, time_axis, delay_grid, window=window)
    dmas_img = dmas(time_signal_magnitude, time_axis, delay_grid, subsample=subsample, window=window)
    return dmas_img * (cf_map ** cf_power), cf_map