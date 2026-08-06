"""
ablation_runner.py

Runs the full 8-variant comparison matrix:
    (Raw DAS, Raw DMAS, DAS+CF, DMAS+CF,
     DAS+bent-ray, DMAS+bent-ray, DAS+bent-ray+CF, DMAS+bent-ray+CF)
across all valid scans, and writes one summary CSV + a per-scan CSV per
variant to results/.

Usage:
    python ablation_runner.py --n-scans 25     # smoke test
    python ablation_runner.py                  # full run (432 scans)
"""

import argparse
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from src.data_loading import load_all_data
from src.pipeline import reconstruct_scan

RESULTS_DIR = Path(__file__).resolve().parent / "results"

VARIANTS = [
    dict(name="Raw DAS",            beamformer="das",  use_bent_ray=False, use_cf=False, use_tvsvd=False),
    dict(name="Raw DMAS",           beamformer="dmas", use_bent_ray=False, use_cf=False, use_tvsvd=False),
    dict(name="DAS+CF",             beamformer="das",  use_bent_ray=False, use_cf=True,  use_tvsvd=False),
    dict(name="DMAS+CF",            beamformer="dmas", use_bent_ray=False, use_cf=True,  use_tvsvd=False),
    dict(name="DAS+BentRay",        beamformer="das",  use_bent_ray=True,  use_cf=False, use_tvsvd=False),
    dict(name="DMAS+BentRay",       beamformer="dmas", use_bent_ray=True,  use_cf=False, use_tvsvd=False),
    dict(name="DAS+BentRay+CF",     beamformer="das",  use_bent_ray=True,  use_cf=True,  use_tvsvd=False),
    dict(name="DMAS+BentRay+CF",    beamformer="dmas", use_bent_ray=True,  use_cf=True,  use_tvsvd=False),
    # TVSVD as its own explicit comparison row, matching the original
    # "SVD-90%+CF" variant (24.65mm on A16F14 — worse than Raw+CF's
    # 10.61mm), rather than silently forced on for every row above.
    dict(name="DAS+CF+TVSVD",       beamformer="das",  use_bent_ray=False, use_cf=True,  use_tvsvd=True),
    dict(name="DAS+BentRay+CF+TVSVD", beamformer="das", use_bent_ray=True, use_cf=True,  use_tvsvd=True),
]


def _run_one(idx, variant, s21, tumor_model):
    try:
        r = reconstruct_scan(
            idx, s21, tumor_model,
            beamformer=variant["beamformer"],
            use_bent_ray=variant["use_bent_ray"],
            use_cf=variant["use_cf"],
            use_tvsvd=variant.get("use_tvsvd", False),
        )
        r.pop("diagnostics", None)
        return ("ok", idx, r)
    except Exception as e:
        return ("fail", idx, str(e))


def run_variant(variant, s21, tumor_model, n_scans, n_jobs=1):
    """
    n_jobs: 1 = serial (safest, easiest to debug). -1 = all CPU cores.
    NOTE: the per-phantom delay-grid cache in pipeline.py is per-process —
    with n_jobs>1 each worker process gets its own cache, so the cache
    speedup and the parallelization speedup both apply but don't multiply
    perfectly (each worker still recomputes a phantom's delay grid the
    first time THAT worker sees it). Still a large combined win over serial
    with no caching.
    """
    outcomes = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_run_one)(idx, variant, s21, tumor_model) for idx in range(n_scans)
    )
    rows = [r for status, idx, r in outcomes if status == "ok"]
    failed = [(idx, r) for status, idx, r in outcomes if status == "fail"]
    df = pd.DataFrame(rows)
    return df, failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-scans", type=int, default=None,
                         help="Limit to first N scans (smoke test). Default: all valid scans.")
    parser.add_argument("--n-jobs", type=int, default=1,
                         help="Parallel workers (joblib). 1=serial, -1=all cores. Default: 1.")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading data...")
    d = load_all_data()
    s21, tumor_model = d["s21"], d["tumor_model"]
    n_scans = args.n_scans or d["n_valid_scans"]
    print(f"Running ablation matrix on {n_scans} scans across {len(VARIANTS)} variants.\n")

    summary_rows = []
    for variant in VARIANTS:
        print(f"Running variant: {variant['name']}")
        df, failed = run_variant(variant, s21, tumor_model, n_scans, n_jobs=args.n_jobs)

        if failed:
            print(f"  {len(failed)} scans failed (e.g. {failed[0]})")

        out_path = RESULTS_DIR / f"ablation_{variant['name'].replace('+', '_')}.csv"
        df.to_csv(out_path, index=False)

        if len(df) > 0:
            summary_rows.append(dict(
                variant=variant["name"],
                n_scans=len(df),
                mean_le_mm=df["localization_error_mm"].mean(),
                median_le_mm=df["localization_error_mm"].median(),
                mean_iou=df["iou"].mean(),
                mean_dice=df["dice"].mean(),
                mean_scr_db=df["scr_db"].mean(),
                detection_rate_20mm=(df["localization_error_mm"] <= 20).mean(),
            ))
        print(f"  -> mean LE: {df['localization_error_mm'].mean():.2f}mm "
              f"| detection@20mm: {(df['localization_error_mm'] <= 20).mean():.1%}\n"
              if len(df) > 0 else "  -> no successful scans\n")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "ablation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()