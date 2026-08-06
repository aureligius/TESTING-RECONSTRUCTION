"""
src/metrics.py

Localization + image-quality metrics, BI-RADS-stratified grouping helpers,
and a front-surface-bias diagnostic (checks whether the predicted-vs-GT
offset is systematically aligned with the antenna-to-tumor direction, which
would confirm front-surface bias rather than random localization error).
"""

import numpy as np
import pandas as pd


def compute_localization_error(peak_x_mm, peak_y_mm, gt_x_mm, gt_y_mm):
    return float(np.sqrt((peak_x_mm - gt_x_mm) ** 2 + (peak_y_mm - gt_y_mm) ** 2))


def compute_iou_dice(tumor_mask, grid_x_mm, grid_y_mm, gt_x_mm, gt_y_mm, gt_r_mm):
    gt_mask = (grid_x_mm - gt_x_mm) ** 2 + (grid_y_mm - gt_y_mm) ** 2 <= gt_r_mm ** 2
    intersection = np.logical_and(tumor_mask, gt_mask).sum()
    union = np.logical_or(tumor_mask, gt_mask).sum()
    iou = intersection / union if union > 0 else 0.0
    denom = tumor_mask.sum() + gt_mask.sum()
    dice = (2 * intersection) / denom if denom > 0 else 0.0
    return float(iou), float(dice)


def compute_scr_cnr(img, tumor_mask):
    """SCR (dB), SMR (dB), CNR — signal region = tumor_mask, clutter = rest."""
    signal_power = img[tumor_mask].mean() if tumor_mask.sum() > 0 else 0.0
    clutter_power = img[~tumor_mask].mean()
    clutter_std = img[~tumor_mask].std()

    scr_db = 10 * np.log10(signal_power / (clutter_power + 1e-12)) if signal_power > 0 else 0.0
    smr_db = 10 * np.log10(signal_power / (img.mean() + 1e-12)) if signal_power > 0 else 0.0
    cnr = (signal_power - clutter_power) / (clutter_std + 1e-12)

    return float(scr_db), float(smr_db), float(cnr)


def compute_all_metrics(img, tumor_mask, x_axis_mm, y_axis_mm, peak_x_mm, peak_y_mm,
                         gt_x_mm, gt_y_mm, gt_r_mm):
    grid_x_mm, grid_y_mm = np.meshgrid(x_axis_mm, y_axis_mm)
    le = compute_localization_error(peak_x_mm, peak_y_mm, gt_x_mm, gt_y_mm)
    iou, dice = compute_iou_dice(tumor_mask, grid_x_mm, grid_y_mm, gt_x_mm, gt_y_mm, gt_r_mm)
    scr_db, smr_db, cnr = compute_scr_cnr(img, tumor_mask)
    return dict(localization_error_mm=le, iou=iou, dice=dice,
                scr_db=scr_db, smr_db=smr_db, cnr=cnr)


# ============================================================================
# BI-RADS-stratified grouping
# ============================================================================
def birads_stratified_summary(results_df, birads_col="birads",
                                metric_cols=("localization_error_mm", "iou", "dice", "scr_db")):
    """results_df must have a birads column (1-4) alongside the metric columns."""
    return results_df.groupby(birads_col)[list(metric_cols)].agg(["mean", "std", "count"])


# ============================================================================
# Front-surface-bias diagnostic
# ============================================================================
def front_surface_bias_diagnostic(results_df):
    """
    Checks whether the predicted-minus-GT offset vector is systematically
    aligned with the direction from the antenna array center (assumed at
    the origin) toward the true tumor location — the signature of
    front-surface bias (peak pulled toward the tumor's near/antenna-facing
    edge) as opposed to unbiased random localization error.

    Requires columns: peak_x_mm, peak_y_mm, gt_x_mm, gt_y_mm in results_df.
    Adds a column 'radial_bias_mm': positive means the predicted peak sits
    CLOSER to the antenna array than the true tumor center (consistent with
    front-surface bias); negative means it overshoots past the tumor.

    Returns (results_df_with_column, summary_dict) where summary_dict has
    mean/std of radial_bias_mm and a fraction-positive statistic — a value
    consistently > 0 across most scans is evidence for front-surface bias
    rather than random error (which would center around 0 with no
    systematic sign).
    """
    df = results_df.copy()

    gt_dist_from_origin = np.sqrt(df["gt_x_mm"] ** 2 + df["gt_y_mm"] ** 2)
    pred_dist_from_origin = np.sqrt(df["peak_x_mm"] ** 2 + df["peak_y_mm"] ** 2)

    # positive = predicted peak is nearer the antenna array (radially closer
    # to origin) than the true tumor center -> consistent with front-surface
    # bias, where the near-side reflection dominates.
    df["radial_bias_mm"] = gt_dist_from_origin - pred_dist_from_origin

    valid = df["radial_bias_mm"].notna()
    summary = dict(
        mean_radial_bias_mm=float(df.loc[valid, "radial_bias_mm"].mean()),
        std_radial_bias_mm=float(df.loc[valid, "radial_bias_mm"].std()),
        fraction_positive=float((df.loc[valid, "radial_bias_mm"] > 0).mean()),
        n_scans=int(valid.sum()),
    )
    return df, summary