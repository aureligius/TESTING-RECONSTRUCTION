"""
select_phantoms.py

Selects 30 phantoms via STRATIFIED sampling on breast size (tertiles) x
BI-RADS density (1-4), from the phantoms that already have verified real
shell-radius/composition data (tumor_model already excludes the ~141
unreliable rows and the data-incomplete A1 family, since those fail the
breast_radius_mm/wave_velocity notna checks in data_loading.py).

CRITICAL: this script does NOT run any reconstruction and does NOT look at
localization error. Selection is based ONLY on phantom-level metadata
(breast_radius_mm, birads) — decided before any performance number exists,
specifically to avoid selection bias (see chat discussion). Do not modify
this script to filter by LE/SCR after the fact.

Usage:
    python select_phantoms.py                # default: 30 phantoms, seed=42
    python select_phantoms.py --n 20 --seed 7
"""

import argparse

import numpy as np
import pandas as pd

from src.data_loading import load_all_data


def build_phantom_table(tumor_model):
    """One row per unique phant_id, with its breast_radius_mm and birads.
    Both are properties of the phantom itself (same for every scan sharing
    that phant_id), so a simple first() per group is exact, not an
    approximation."""
    return (
        tumor_model.groupby("phant_id")
        .agg(breast_radius_mm=("breast_radius_mm", "first"), birads=("birads", "first"))
        .reset_index()
    )


def stratified_select(phantom_table, n=30, seed=42):
    phantom_table = phantom_table.copy()

    # Size tertiles based on the ACTUAL observed distribution, not arbitrary
    # fixed cutoffs — keeps strata roughly balanced regardless of how the
    # dataset's phantom sizes happen to be spread.
    phantom_table["size_tier"] = pd.qcut(
        phantom_table["breast_radius_mm"], q=3, labels=["small", "medium", "large"]
    )
    phantom_table["stratum"] = (
        phantom_table["size_tier"].astype(str) + "_birads" + phantom_table["birads"].astype(str)
    )

    rng = np.random.default_rng(seed)
    strata = phantom_table["stratum"].unique()

    # Proportional allocation (largest-remainder method) so bigger strata
    # get more slots, but every non-empty stratum gets at least a shot.
    counts = phantom_table["stratum"].value_counts()
    raw_alloc = counts / counts.sum() * n
    base_alloc = np.floor(raw_alloc).astype(int)
    remainder = n - base_alloc.sum()
    remainders = (raw_alloc - base_alloc).sort_values(ascending=False)
    for stratum in remainders.index[:remainder]:
        base_alloc[stratum] += 1

    selected_ids = []
    for stratum in strata:
        group = phantom_table[phantom_table["stratum"] == stratum]
        take = min(base_alloc.get(stratum, 0), len(group))
        chosen = rng.choice(group["phant_id"].values, size=take, replace=False)
        selected_ids.extend(chosen.tolist())

    # If proportional allocation came up short (small strata rounding down),
    # top up randomly from whatever's left, still seeded/deterministic.
    if len(selected_ids) < n:
        remaining_pool = phantom_table[~phantom_table["phant_id"].isin(selected_ids)]
        top_up = rng.choice(
            remaining_pool["phant_id"].values,
            size=min(n - len(selected_ids), len(remaining_pool)),
            replace=False,
        )
        selected_ids.extend(top_up.tolist())

    return phantom_table[phantom_table["phant_id"].isin(selected_ids)].sort_values(
        ["size_tier", "birads"]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="Number of phantoms to select.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (reproducibility).")
    args = parser.parse_args()

    print("Loading data...")
    d = load_all_data()
    tumor_model = d["tumor_model"]

    phantom_table = build_phantom_table(tumor_model)
    print(f"\nTotal phantoms with verified data: {len(phantom_table)}")
    print(f"Breast radius range: {phantom_table['breast_radius_mm'].min():.1f}mm - "
          f"{phantom_table['breast_radius_mm'].max():.1f}mm")
    print(f"BI-RADS distribution: \n{phantom_table['birads'].value_counts().sort_index()}")

    selected = stratified_select(phantom_table, n=args.n, seed=args.seed)

    print(f"\n{'=' * 70}")
    print(f"SELECTED {len(selected)} PHANTOMS (seed={args.seed}, stratified by size x BI-RADS)")
    print(f"{'=' * 70}")
    print(selected.to_string(index=False))

    print(f"\nStratum coverage:")
    print(selected["stratum"].value_counts().sort_index())

    print(f"\nphant_id list (paste into other scripts):")
    print(selected["phant_id"].tolist())


if __name__ == "__main__":
    main()