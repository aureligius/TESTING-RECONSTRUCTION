"""
train_svm.py

Trains a linear-kernel SVM to rank candidate peaks (real tumor vs false
positive), using GroupKFold cross-validation GROUPED BY PHANT_ID (not
scan_idx) — candidates from different scans of the SAME phantom share
delay-grid geometry and artifact strength, so grouping only by scan_idx
would let the model implicitly learn phantom-specific patterns from one
scan and unfairly "recognize" them in another scan of that same phantom
(see chat discussion).

The actual test that matters is NOT classification accuracy — it's whether
using the SVM's out-of-fold ranking to pick a location, instead of plain
argmax, produces a lower real LE. Everything else (AUC, feature weights) is
secondary/diagnostic.

Usage:
    # first generate the candidate dataset:
    python baseline_argmax_test.py --n-phantoms 30 --n-jobs 4 \\
        --save-candidates svm_candidates.csv
    # then train + evaluate:
    python train_svm.py --candidates svm_candidates.csv
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

FEATURE_COLS = [
    "cf_at_peak", "cf_background_ratio", "area_fraction",
    "dist_to_ring_center_mm", "dist_to_ring_center_ratio",
    "local_scr_db", "hessian_eigenratio",
]
# dist_to_shell_center_mm deliberately excluded — mathematically identical
# to dist_to_ring_center_mm in this dataset (shell_center defaults to
# (0,0)), redundant and would muddy the learned coefficients.

DETECTION_THRESHOLD_MM = 20.0


def run_grouped_cv(df, n_splits=5, seed=42):
    """
    Returns df with an added 'oof_score' column (out-of-fold predicted
    decision-function value — higher = more likely real tumor), computed
    without any candidate ever being scored by a model that saw its own
    phantom during training.
    """
    X = df[FEATURE_COLS].values
    y = df["label"].values
    groups = df["phant_id"].values

    n_groups = df["phant_id"].nunique()
    n_splits = min(n_splits, n_groups)
    gkf = GroupKFold(n_splits=n_splits)

    oof_scores = np.zeros(len(df))
    fold_aucs = []

    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        clf = SVC(kernel="linear", class_weight="balanced",
                  probability=False, random_state=seed)
        clf.fit(X_train, y[train_idx])

        scores = clf.decision_function(X_test)
        oof_scores[test_idx] = scores

        if len(np.unique(y[test_idx])) == 2:
            fold_auc = roc_auc_score(y[test_idx], scores)
            fold_aucs.append(fold_auc)
            print(f"  fold {fold_i + 1}/{n_splits}: "
                  f"{len(train_idx)} train / {len(test_idx)} test candidates, "
                  f"test AUC={fold_auc:.3f}")
        else:
            print(f"  fold {fold_i + 1}/{n_splits}: "
                  f"{len(train_idx)} train / {len(test_idx)} test candidates, "
                  f"AUC undefined (only one class in test fold)")

    df = df.copy()
    df["oof_score"] = oof_scores
    return df, fold_aucs


def direct_le_comparison(df):
    """The test that actually matters: for each scan, pick the candidate
    with the highest out-of-fold SVM score, and compare the resulting LE
    against plain argmax (candidate #0 in each scan's original,
    brightest-first order — same convention used throughout this project)."""
    argmax_les, svm_les = [], []
    for scan_idx, group in df.groupby("scan_idx"):
        group_sorted = group.sort_index()   # restore original append order
        argmax_les.append(group_sorted.iloc[0]["le_to_gt_mm"])
        svm_les.append(group.loc[group["oof_score"].idxmax(), "le_to_gt_mm"])

    argmax_les = pd.Series(argmax_les)
    svm_les = pd.Series(svm_les)

    print(f"\n{'=' * 70}")
    print(f"DIRECT LE TEST: SVM-ranked vs plain argmax (n={len(argmax_les)} scans)")
    print(f"{'=' * 70}")
    for name, series in [("Plain argmax (candidate #0)", argmax_les),
                          ("SVM-ranked (out-of-fold, highest score wins)", svm_les)]:
        print(f"  {name:<46}: mean LE={series.mean():.2f}mm  "
              f"median LE={series.median():.2f}mm  "
              f"detection@20mm={(series <= DETECTION_THRESHOLD_MM).mean():.1%}")
    print(f"  (this is the real test — everything else in this script is "
          f"diagnostic. If SVM-ranked LE isn't meaningfully lower, the "
          f"trained classifier doesn't beat argmax either, same as every "
          f"hand-built heuristic tried before it — see chat discussion)")


def print_feature_weights(df, seed=42):
    """Trains ONE final model on ALL data (not for evaluation — CV already
    did that honestly — just to report interpretable coefficients for the
    paper). Linear-kernel SVM coefficients are directly interpretable:
    sign and magnitude show each feature's learned contribution."""
    X = df[FEATURE_COLS].values
    y = df["label"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = SVC(kernel="linear", class_weight="balanced", random_state=seed)
    clf.fit(X_scaled, y)

    print(f"\n{'=' * 70}")
    print(f"LEARNED FEATURE WEIGHTS (linear SVM, fit on ALL data — for the "
          f"paper's interpretability discussion, NOT used for the LE test above)")
    print(f"{'=' * 70}")
    weights = list(zip(FEATURE_COLS, clf.coef_[0]))
    weights.sort(key=lambda x: abs(x[1]), reverse=True)
    for name, w in weights:
        direction = "favors REAL tumor" if w > 0 else "favors FALSE POSITIVE"
        print(f"  {name:<28}: weight={w:+.3f}  ({direction})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str, default="svm_candidates.csv",
                         help="Path to the candidate CSV from baseline_argmax_test.py "
                              "--save-candidates.")
    parser.add_argument("--n-splits", type=int, default=5,
                         help="Number of GroupKFold splits (grouped by phant_id).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading candidates from {args.candidates}...")
    df = pd.read_csv(args.candidates)
    print(f"  {len(df)} candidates, {df['scan_idx'].nunique()} scans, "
          f"{df['phant_id'].nunique()} phantoms")
    print(f"  {df['label'].sum()} positive ({df['label'].mean():.1%}), "
          f"{(df['label'] == 0).sum()} negative")

    print(f"\nRunning {args.n_splits}-fold GroupKFold CV (grouped by phant_id, "
          f"NOT scan_idx — see module docstring)...")
    df, fold_aucs = run_grouped_cv(df, n_splits=args.n_splits, seed=args.seed)

    if fold_aucs:
        print(f"\nMean out-of-fold AUC across folds: {np.mean(fold_aucs):.3f} "
              f"(std={np.std(fold_aucs):.3f})")
    overall_auc = roc_auc_score(df["label"], df["oof_score"])
    print(f"Overall out-of-fold AUC (all folds pooled): {overall_auc:.3f}")

    direct_le_comparison(df)
    print_feature_weights(df, seed=args.seed)


if __name__ == "__main__":
    main()