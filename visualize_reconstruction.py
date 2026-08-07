"""
visualize_reconstructions.py

Generates ACTUAL reconstructed images (not just numbers) for a handful of
representative scans — addressing the gap where this entire investigation
computed images internally but never rendered/saved/displayed one, per an
earlier explicit "just show me the LE number" request for speed.

For each selected scan, saves a 3-panel PNG: Raw DAS, DAS+CF,
DAS+CF-debiased-zscore, each with ground truth (green X) and the argmax
peak (red +) marked, plus LE printed in the panel title. This directly
shows the ring-center artifact and, for boundary cases, the SCR collapse —
visually, not just as numbers.

Usage:
    python visualize_reconstructions.py --n-phantoms 30
    python visualize_reconstructions.py --scan-idx 143   # a specific scan
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loading import load_all_data
from src import physics
from src import signal_processing as sp
from src import beamforming as bf
from select_phantoms import build_phantom_table, stratified_select
from baseline_argmax_test import (
    build_grid, get_cf_baseline_cached, debias_cf_zscore, argmax_localize_and_score,
)


def reconstruct_images_for_scan(scan_idx, s21, tumor_model, id_to_original_idx):
    """Minimal version of the reconstruction pipeline — just enough to get
    the three images for plotting, reusing the same physics/beamforming
    functions used everywhere else in this project."""
    row = tumor_model.iloc[scan_idx]
    breast_radius_mm = float(row["breast_radius_mm"])
    fat_frac = float(row["fat_fraction"])
    fib_frac = float(row["fib_fraction"])
    v_tissue, _ = physics.compute_tissue_velocity(fat_frac, fib_frac)

    ant_rad_mm = float(row.get("ant_rad", 21.5)) * 10.0
    geom = physics.get_antenna_geometry(ant_rad_mm)

    s21_idx = int(row["original_s21_idx"])
    fd_scan = s21[s21_idx]

    # Empty-chamber calibration subtraction — was MISSING from this script
    # until now (see chat discussion: this is why the center artifact looked
    # far more dominant here than in baseline_argmax_test.py's numbers,
    # which always had this applied). Matches baseline_argmax_test.py's
    # reconstruct_and_score_all_variants() exactly.
    emp_ref_id = row.get("emp_ref_id", None)
    if emp_ref_id is not None and not pd.isna(emp_ref_id) and int(emp_ref_id) in id_to_original_idx:
        emp_idx = id_to_original_idx[int(emp_ref_id)]
        fd_scan = fd_scan - s21[emp_idx]

    time_signal = sp.to_time_domain(fd_scan)
    time_axis = sp.get_time_axis(time_signal.shape[0])
    td_mag = np.abs(time_signal)

    grid_x_mm, grid_y_mm, axis_mm, grid_radius_mm = build_grid(breast_radius_mm)
    grid_x_m, grid_y_m = grid_x_mm.ravel() / 1000.0, grid_y_mm.ravel() / 1000.0

    delay_grid = physics.two_medium_delay(
        geom["ant_x"], geom["ant_y"], geom["ant_x_b"], geom["ant_y_b"],
        grid_x_m, grid_y_m, breast_radius_mm / 1000.0, v_tissue,
    )
    delay_grid = delay_grid.reshape(-1, *grid_x_mm.shape)

    raw_das = bf.das_coherent(time_signal, time_axis, delay_grid)
    das_image_raw, cf_map, das_cf_img = bf.das_coherent_cf(time_signal, time_axis, delay_grid)

    n_ant = time_signal.shape[1]
    baseline_mean, baseline_std = get_cf_baseline_cached(
        row["phant_id"], delay_grid, time_axis, n_ant)
    cf_zscore = debias_cf_zscore(cf_map, baseline_mean, baseline_std)
    das_debiased = das_image_raw * cf_zscore

    gt_x_mm, gt_y_mm = float(row["tumor_x_mm"]), float(row["tumor_y_mm"])

    return dict(
        images={"Raw DAS": raw_das, "DAS+CF": das_cf_img, "DAS+CF-debiased-zscore": das_debiased},
        axis_mm=axis_mm, grid_radius_mm=grid_radius_mm,
        gt_x_mm=gt_x_mm, gt_y_mm=gt_y_mm,
        phant_id=row["phant_id"], breast_radius_mm=breast_radius_mm,
    )


def plot_scan(scan_idx, result, out_dir=".", display_margin=None):
    """
    display_margin: if given (e.g. 1.2), crops the DISPLAYED view to
    breast_radius_mm * display_margin via ax.set_xlim/ylim — does NOT
    change the underlying computed grid (imshow's `extent` stays the full
    GRID_MARGIN_FACTOR=1.5 grid from build_grid() in baseline_argmax_test.py).
    This is purely cosmetic — LE/SCR and every other number this project has
    reported are computed from the full grid and are unaffected by this.
    If display_margin=None (default), shows the full computed grid as-is.
    """
    images, axis_mm, grid_radius_mm = result["images"], result["axis_mm"], result["grid_radius_mm"]
    gt_x_mm, gt_y_mm = result["gt_x_mm"], result["gt_y_mm"]
    breast_radius_mm = result["breast_radius_mm"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    extent = [-grid_radius_mm, grid_radius_mm, -grid_radius_mm, grid_radius_mm]

    for ax, (name, img) in zip(axes, images.items()):
        le, scr, on_edge, px, py = argmax_localize_and_score(
            img, axis_mm, grid_radius_mm, gt_x_mm, gt_y_mm)
        im = ax.imshow(img, extent=extent, origin="lower", cmap="turbo", aspect="equal")
        ax.plot(gt_x_mm, gt_y_mm, "gx", markersize=14, markeredgewidth=3, label="Ground truth")
        ax.plot(px, py, "r+", markersize=14, markeredgewidth=3, label="Predicted (argmax)")
        ax.set_title(f"{name}\nLE={le:.1f}mm  SCR={scr:.1f}dB", fontsize=11)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.legend(loc="upper right", fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.8)

        if display_margin is not None:
            display_radius_mm = breast_radius_mm * display_margin
            ax.set_xlim(-display_radius_mm, display_radius_mm)
            ax.set_ylim(-display_radius_mm, display_radius_mm)

    fig.suptitle(f"scan_idx={scan_idx}  phant_id={result['phant_id']}  "
                 f"breast_radius={result['breast_radius_mm']:.1f}mm", fontsize=13)
    plt.tight_layout()
    out_path = f"{out_dir}/reconstruction_scan_{scan_idx}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-phantoms", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-viz", type=int, default=10,
                         help="Number of scans to visualize when --scan-idx isn't given "
                              "(one per distinct phantom, seeded). Default: 10.")
    parser.add_argument("--scan-idx", type=int, default=None,
                         help="Visualize one specific scan_idx instead of the default set.")
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--display-margin", type=float, default=None,
                         help="Crop the DISPLAYED view to breast_radius_mm * this value "
                              "(e.g. 1.2). Does NOT change the underlying computed grid "
                              "(still GRID_MARGIN_FACTOR=1.5 from baseline_argmax_test.py) "
                              "or any LE/SCR number — display only. Default: show full grid.")
    args = parser.parse_args()

    print("Loading data...")
    d = load_all_data()
    s21, tumor_model = d["s21"], d["tumor_model"]
    id_to_original_idx = d["id_to_original_idx"]

    if args.n_phantoms > 0:
        phantom_table = build_phantom_table(tumor_model)
        selected = stratified_select(phantom_table, n=args.n_phantoms, seed=args.seed)
        selected_ids = selected["phant_id"].tolist()
        tumor_model = tumor_model[tumor_model["phant_id"].isin(selected_ids)].reset_index(drop=True)

    if args.scan_idx is not None:
        scan_indices = [args.scan_idx]
    else:
        # One scan per DISTINCT phantom (same logic as --one-per-phantom in
        # baseline_argmax_test.py) — up to --n-viz phantoms, seeded/
        # reproducible — instead of the old hardcoded [0,1,2,143] list.
        rng = np.random.default_rng(args.seed)
        scan_indices = []
        for pid, group in tumor_model.groupby("phant_id"):
            scan_indices.append(int(rng.choice(group.index.values)))
        rng.shuffle(scan_indices)
        scan_indices = sorted(scan_indices[:args.n_viz])
        print(f"No --scan-idx given — visualizing {len(scan_indices)} scans, "
              f"one per distinct phantom: {scan_indices}")

    for scan_idx in scan_indices:
        if scan_idx >= len(tumor_model):
            print(f"Skipping scan_idx={scan_idx} — out of range for this tumor_model "
                  f"(len={len(tumor_model)})")
            continue
        print(f"Reconstructing scan_idx={scan_idx}...")
        result = reconstruct_images_for_scan(scan_idx, s21, tumor_model, id_to_original_idx)
        plot_scan(scan_idx, result, out_dir=args.out_dir, display_margin=args.display_margin)


if __name__ == "__main__":
    main()