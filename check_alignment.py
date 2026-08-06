"""
check_alignment.py — one-off diagnostic. Answers two questions before we
build the calibration-subtraction fix:

  1. For empty-chamber reference scans, is phant_id stored as NaN or as a
     literal empty string ""? This determines whether data_loading.py's
     `metadata[metadata["phant_id"] != ""]` filter silently drops rows
     (misaligning metadata position with s21 array position) or correctly
     leaves them in place (since NaN != "" is True, so NaN rows survive
     that filter).

  2. Does raw, completely unfiltered metadata have exactly 1301 rows
     (matching s21.shape[0]) with unique 'id' values, confirming 1:1
     positional alignment is safe to rely on.

Run this once, no other files touched.
"""

import pickle
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

with open(DATA_DIR / "fd_data_gen_two_s21.pickle", "rb") as f:
    s21 = pickle.load(f)

raw_metadata = pd.read_pickle(DATA_DIR / "metadata_gen_two.pickle")
raw_metadata = pd.DataFrame(raw_metadata)

print("=" * 60)
print("BASIC COUNTS")
print("=" * 60)
print(f"s21.shape[0]        : {s21.shape[0]}")
print(f"raw_metadata rows   : {len(raw_metadata)}")
print(f"Match?              : {s21.shape[0] == len(raw_metadata)}")

print("\n" + "=" * 60)
print("phant_id NULL-NESS CHECK")
print("=" * 60)
n_nan = raw_metadata["phant_id"].isna().sum()
n_empty_str = (raw_metadata["phant_id"].astype(str).str.strip() == "").sum()
print(f"Rows where phant_id is NaN (pandas-native)      : {n_nan}")
print(f"Rows where phant_id.astype(str).strip() == ''   : {n_empty_str}")
print("(152 expected empty-chamber scans per earlier explore.py output)")

print("\n" + "=" * 60)
print("'id' FIELD UNIQUENESS + emp_ref_id PRESENCE")
print("=" * 60)
has_id = "id" in raw_metadata.columns
has_emp_ref = "emp_ref_id" in raw_metadata.columns
print(f"'id' column present         : {has_id}")
print(f"'emp_ref_id' column present : {has_emp_ref}")
if has_id:
    print(f"Unique id values            : {raw_metadata['id'].nunique()} / {len(raw_metadata)}")

print("\n" + "=" * 60)
print("WHAT data_loading.load_metadata() WOULD DROP")
print("=" * 60)
would_keep = raw_metadata[raw_metadata["phant_id"].astype(str).str.strip() != ""]
n_dropped = len(raw_metadata) - len(would_keep)
print(f"Rows load_metadata() would DROP : {n_dropped}")
if n_dropped > 0:
    print("*** ALIGNMENT RISK CONFIRMED: rows are being dropped before merge,")
    print("*** which shifts row position relative to s21 for everything after")
    print("*** the first dropped row. The shared pipeline needs a fix.")
else:
    print("No rows dropped — metadata position stays aligned with s21 position.")
    print("The alignment risk does NOT apply; existing pipeline indexing is safe.")