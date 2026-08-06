"""
src/signal_processing.py

Frequency->time domain transform (ICZT, 1-8GHz per UM-BMID spec — NOT plain
IFFT with a 2-9GHz axis) and Ursula's hybrid TVSVD clutter suppression
(energy elbow AND spatial-uniformity check, so a strong tumor-bearing
component isn't discarded alongside genuine skin clutter just because it's
also energetic).
"""

import numpy as np
from scipy.signal.windows import tukey

try:
    from umbmid.sigproc import iczt
    ICZT_AVAILABLE = True
except ImportError:
    ICZT_AVAILABLE = False

# Confirmed against UM-BMID Gen-2 docs (not Ursula's original 2-9GHz assumption)
FREQ_START_HZ = 1e9
FREQ_STOP_HZ = 8e9
TIME_START_S = 0.0
TIME_STOP_S = 6e-9
N_TIME_PTS = 1024


def to_time_domain(fd_signal, window_alpha=0.25, n_time_pts=N_TIME_PTS):
    """
    fd_signal: (n_freq, n_ant) complex frequency-domain signal for one scan.
    Returns (n_time_pts, n_ant) complex time-domain signal via ICZT over the
    confirmed 1-8GHz / 0-6ns window.
    """
    if not ICZT_AVAILABLE:
        raise ImportError(
            "umbmid.sigproc.iczt not importable — copy the umbmid/ package "
            "into the repo root (see README) before running the pipeline."
        )
    window = tukey(fd_signal.shape[0], alpha=window_alpha)
    fd_windowed = fd_signal * window[:, None]
    return iczt(fd_windowed, ini_t=TIME_START_S, fin_t=TIME_STOP_S,
                n_time_pts=n_time_pts, ini_f=FREQ_START_HZ, fin_f=FREQ_STOP_HZ)


def get_time_axis(n_time_pts=N_TIME_PTS):
    return np.linspace(TIME_START_S, TIME_STOP_S, n_time_pts)


# ============================================================================
# Hybrid TVSVD (Ursula's Phase 5.5 approach)
# ============================================================================
def select_tvsvd_rank_adaptive(S, min_rank=1, max_energy=0.98):
    """Kneedle elbow detection on cumulative-energy curve. Falls back to a
    fixed 90% energy cut if the curve is too flat/short to have a clear knee."""
    energy = S ** 2
    cum = np.cumsum(energy) / np.sum(energy)
    n = len(cum)
    if n < 3:
        return max(min_rank, int(np.argmax(cum >= 0.90)))

    x_norm = np.arange(n) / (n - 1)
    p1 = np.array([x_norm[0], cum[0]])
    p2 = np.array([x_norm[-1], cum[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-12:
        return max(min_rank, int(np.argmax(cum >= 0.90)))

    line_unit = line_vec / line_len
    pts = np.stack([x_norm, cum], axis=1) - p1
    proj_len = pts @ line_unit
    proj_pts = np.outer(proj_len, line_unit)
    dist = np.linalg.norm(pts - proj_pts, axis=1)

    knee = max(min_rank, int(np.argmax(dist)))
    hard_cap = int(np.argmax(cum >= max_energy))
    return min(knee, hard_cap) if hard_cap > 0 else knee


def select_tvsvd_rank_hybrid(S, Vt, energy_lower=0.01, flatness_thresh=0.85,
                              max_energy=0.98):
    """
    A component is removed only if it is BOTH energetic (above the knee-
    selected rank and above energy_lower fraction) AND spatially uniform
    across antennas (Vt row flatness close to 1 — the signature of a shared
    skin reflection, not a real off-center tumor return).

    Returns a boolean mask (True = remove), not just a rank cutoff, so a
    strong-but-non-uniform tumor component surviving inside the "knee" range
    is protected instead of discarded.
    """
    knee = select_tvsvd_rank_adaptive(S, max_energy=max_energy)
    energy_frac = (S ** 2) / np.sum(S ** 2)

    remove_mask = np.zeros(len(S), dtype=bool)
    for k in range(knee):
        row = Vt[k]
        flatness = np.abs(np.mean(row)) / (np.sqrt(np.mean(row ** 2)) + 1e-12)
        if energy_frac[k] > energy_lower and flatness > flatness_thresh:
            remove_mask[k] = True
    return remove_mask


def apply_hybrid_tvsvd(time_signal):
    """time_signal: (n_time, n_ant) complex. Returns (filtered_signal, n_removed)."""
    U, S, Vt = np.linalg.svd(time_signal, full_matrices=False)
    remove_mask = select_tvsvd_rank_hybrid(S, Vt)
    S_filtered = S.copy()
    S_filtered[remove_mask] = 0.0
    filtered = U @ np.diag(S_filtered) @ Vt
    return filtered, int(remove_mask.sum())