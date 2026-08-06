"""
src/data_loading.py

Loads the UM-BMID Gen-2 S21 dataset, metadata, and Ursula's phantom_database.csv,
merges them, and backfills any phantom missing from the CSV using Aurel's
hardcoded real-measurement tables (from the phantomInfo/phantomBIRADS source
tables) rather than dropping those scans.

Also converts Ursula's density_class column ("C1".."C4") into an integer
BI-RADS category (1-4) for stratified analysis.

Usage:
    from src.data_loading import load_all_data
    d = load_all_data()
    d['s21']            # complex128 array (n_scans, n_freq, 72)
    d['tumor_model']     # DataFrame, one row per usable tumor-bearing scan
"""

import re
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

S21_PATH = DATA_DIR / "fd_data_gen_two_s21.pickle"
META_PATH = DATA_DIR / "metadata_gen_two.pickle"
PHANTOM_CSV_PATH = DATA_DIR / "phantom_database.csv"

C_LIGHT = 3e8
EPS_FAT = 7.0    # glycerin mixture (adipose-mimicking)
EPS_FIB = 45.0   # Triton X-100 mixture (fibroglandular-mimicking)

# ============================================================================
# Aurel's real-measurement fallback tables (phantomInfo / phantomBIRADS
# source tables). Used ONLY when a phantom is missing from Ursula's
# phantom_database.csv, so we don't silently drop scans that the CSV's
# cleaning step happened to filter out.
# ============================================================================
ADIPOSE_SHELL_RADIUS_ANTENNA_PLANE_MM = {
    'A1': 22.9, 'A2': 48.5, 'A3': 56.9, 'A11': 33.1, 'A12': 39.0,
    'A13': 43.8, 'A14': 45.4, 'A15': 56.2, 'A16': 57.0,
}

PHANTOM_COMPOSITION_MM3 = {
    'A1F1': (294710, 12829), 'A1F3': (294710, 58759), 'A1F4': (294710, 108762),
    'A1F11': (294710, 80366),
    'A2F1': (729324, 12829), 'A2F2': (729324, 29686), 'A2F3': (729324, 58759),
    'A2F4': (729324, 108762), 'A2F5': (729324, 275624), 'A2F11': (729324, 80366),
    'A2F12': (729324, 130177), 'A2F13': (729324, 148442), 'A2F14': (729324, 175302),
    'A3F1': (1113320, 12829), 'A3F2': (1113320, 29686), 'A3F3': (1113320, 58759),
    'A3F4': (1113320, 108762), 'A3F5': (1113320, 275624), 'A3F11': (1113320, 80366),
    'A3F12': (1113320, 130177), 'A3F13': (1113320, 148442), 'A3F14': (1113320, 175302),
    'A11F1': (458352, 12829), 'A11F3': (458352, 58759), 'A11F4': (458352, 108762),
    'A11F11': (458352, 80366), 'A11F12': (458352, 130177), 'A11F14': (458352, 175302),
    'A12F1': (567850, 12829), 'A12F2': (567850, 29686), 'A12F3': (567850, 58759),
    'A12F11': (567850, 80366), 'A12F13': (567850, 148442),
    'A13F1': (652080, 12829), 'A13F2': (652080, 29686), 'A13F3': (652080, 58759),
    'A13F4': (652080, 108762), 'A13F11': (652080, 80366), 'A13F12': (652080, 130177),
    'A13F14': (652080, 175302),
    'A14F1': (713177, 12829), 'A14F2': (713177, 29686), 'A14F3': (713177, 58759),
    'A14F4': (713177, 108762), 'A14F11': (713177, 80366), 'A14F12': (713177, 130177),
    'A14F13': (713177, 148442), 'A14F14': (713177, 175302),
    'A15F1': (1028750, 12829), 'A15F2': (1028750, 29686), 'A15F3': (1028750, 58759),
    'A15F4': (1028750, 108762), 'A15F5': (1028750, 275624), 'A15F11': (1028750, 80366),
    'A15F12': (1028750, 130177), 'A15F13': (1028750, 148442), 'A15F14': (1028750, 175302),
    'A16F1': (1034420, 12829), 'A16F2': (1034420, 29686), 'A16F3': (1034420, 58759),
    'A16F4': (1034420, 108762), 'A16F5': (1034420, 275624), 'A16F11': (1034420, 80366),
    'A16F12': (1034420, 130177), 'A16F13': (1034420, 148442), 'A16F14': (1034420, 175302),
}


def _parse_adipose_id(phant_id):
    """'A16F14' -> 'A16'. Returns None if the pattern doesn't match."""
    m = re.match(r'^(A\d+)F\d+$', str(phant_id))
    return m.group(1) if m else None


def _density_class_to_birads(density_class):
    """'C1'..'C4' -> 1..4 (int). Returns NaN if not in the expected format."""
    if pd.isna(density_class):
        return np.nan
    m = re.match(r'^C(\d)$', str(density_class).strip())
    return int(m.group(1)) if m else np.nan


def load_s21(path=S21_PATH):
    with open(path, "rb") as f:
        s21 = pickle.load(f)
    assert np.iscomplexobj(s21), "S21 dataset is not complex"
    assert np.all(np.isfinite(s21)), "S21 dataset has NaN/Inf"
    return s21


def load_metadata(path=META_PATH):
    metadata = pd.read_pickle(path)
    metadata = pd.DataFrame(metadata).copy()

    # Stamp each row's position in the RAW, unfiltered array BEFORE any
    # filtering or reset_index happens. This is a data column, not the
    # DataFrame's index, so it survives every downstream filter/reset and
    # lets later code correctly index into s21 (whose row order matches
    # this raw, unfiltered order) even after rows have been dropped and
    # the DataFrame's own index has been reset. Without this, scan_idx
    # positions in the filtered/merged DataFrame silently stop
    # corresponding to the same position in s21.
    metadata["original_s21_idx"] = np.arange(len(metadata))

    metadata["phant_id"] = metadata["phant_id"].astype(str).str.strip()
    metadata = metadata[metadata["phant_id"] != ""].reset_index(drop=True)
    return metadata


def build_id_to_original_idx(path=META_PATH):
    """
    {id: original_s21_idx} for ALL 1301 raw rows, including empty-chamber
    reference scans that load_metadata() filters out (phant_id == "").
    Needed for empty-chamber calibration subtraction, since a scan's
    emp_ref_id points at one of those filtered-out rows — load_metadata()'s
    output alone can't resolve it.
    """
    raw = pd.read_pickle(path)
    raw = pd.DataFrame(raw)
    return dict(zip(raw["id"], np.arange(len(raw))))


def load_phantom_db(path=PHANTOM_CSV_PATH):
    return pd.read_csv(path)


def merge_and_backfill(metadata, phantom_db):
    """
    Merge metadata with phantom_database.csv (Ursula's source, preferred),
    then backfill any row where the CSV merge produced NaN shell_radius /
    fib_volume using Aurel's hardcoded real-measurement tables, keyed off
    the adipose ID parsed from phant_id.

    Returns the merged DataFrame with columns renamed for downstream use:
    breast_radius_mm, tumor_radius_mm, tumor_x_mm, tumor_y_mm, birads.
    """
    merged = metadata.merge(
        phantom_db, left_on="phant_id", right_on="phantom_id", how="left"
    )

    n_missing_before = merged["shell_radius"].isna().sum()
    if n_missing_before > 0:
        print(f"[data_loading] {n_missing_before} rows missing phantom_database.csv "
              f"match — attempting fallback-table backfill...")

    for idx in merged.index[merged["shell_radius"].isna()]:
        adi_id = _parse_adipose_id(merged.at[idx, "phant_id"])
        fallback_radius = ADIPOSE_SHELL_RADIUS_ANTENNA_PLANE_MM.get(adi_id)
        if fallback_radius is not None:
            merged.at[idx, "shell_radius"] = fallback_radius

        adi_vol, fib_vol = PHANTOM_COMPOSITION_MM3.get(
            str(merged.at[idx, "phant_id"]), (None, None)
        )
        if adi_vol is not None:
            merged.at[idx, "shell_volume"] = adi_vol
            merged.at[idx, "fib_volume"] = fib_vol

    n_missing_after = merged["shell_radius"].isna().sum()
    if n_missing_before > 0:
        print(f"[data_loading] backfill recovered {n_missing_before - n_missing_after} "
              f"rows; {n_missing_after} still unmatched (will be dropped downstream).")

    # tissue fractions + effective permittivity
    merged["fat_volume"] = merged["shell_volume"] - merged["fib_volume"]
    merged["fat_fraction"] = merged["fat_volume"] / merged["shell_volume"]
    merged["fib_fraction"] = merged["fib_volume"] / merged["shell_volume"]
    merged["eps_eff"] = merged["fat_fraction"] * EPS_FAT + merged["fib_fraction"] * EPS_FIB
    merged["wave_velocity"] = C_LIGHT / np.sqrt(merged["eps_eff"])

    # BI-RADS from density_class ("C1".."C4" -> 1..4)
    if "density_class" in merged.columns:
        merged["birads"] = merged["density_class"].apply(_density_class_to_birads)
    else:
        merged["birads"] = np.nan

    # rename for downstream consistency
    merged = merged.rename(columns={
        "shell_radius": "breast_radius_mm",
        "tum_rad": "tumor_radius_mm",
        "tum_x": "tumor_x_mm",
        "tum_y": "tumor_y_mm",
    })

    # unit check: UM-BMID tumor radii are 5/10/15/20mm. If max < 2.0, values
    # are actually in cm and need scaling.
    if merged["tumor_radius_mm"].notna().any() and merged["tumor_radius_mm"].max() < 2.0:
        print("[data_loading] tumor columns detected in cm -> converting to mm")
        merged["tumor_radius_mm"] *= 10.0
        merged["tumor_x_mm"] *= 10.0
        merged["tumor_y_mm"] *= 10.0

    return merged


def build_tumor_model(physical_model):
    """Filter to scans with complete tumor + physical-model data."""
    tumor_model = physical_model[
        physical_model["tumor_radius_mm"].notna()
        & physical_model["tumor_x_mm"].notna()
        & physical_model["tumor_y_mm"].notna()
        & physical_model["breast_radius_mm"].notna()
        & physical_model["wave_velocity"].notna()
    ].reset_index(drop=True)
    return tumor_model


def load_all_data(s21_path=S21_PATH, meta_path=META_PATH, csv_path=PHANTOM_CSV_PATH):
    """One-call load + merge + backfill + filter. Returns a dict of everything
    downstream modules need."""
    s21 = load_s21(s21_path)
    metadata = load_metadata(meta_path)
    phantom_db = load_phantom_db(csv_path)
    id_to_original_idx = build_id_to_original_idx(meta_path)

    physical_model = merge_and_backfill(metadata, phantom_db)
    tumor_model = build_tumor_model(physical_model)

    n_valid = min(len(tumor_model), s21.shape[0])

    print(f"[data_loading] s21 scans: {s21.shape[0]}  |  "
          f"tumor_model rows: {len(tumor_model)}  |  usable range: 0..{n_valid - 1}")
    if "birads" in tumor_model.columns:
        counts = tumor_model["birads"].value_counts().sort_index()
        print(f"[data_loading] BI-RADS distribution among usable tumor scans:\n{counts}")

    return dict(
        s21=s21,
        metadata=metadata,
        phantom_db=phantom_db,
        physical_model=physical_model,
        tumor_model=tumor_model,
        n_valid_scans=n_valid,
        id_to_original_idx=id_to_original_idx,
    )


if __name__ == "__main__":
    d = load_all_data()
    print(d["tumor_model"].head())