#!/usr/bin/env python3
"""
D_discretization_portable.py

Minimal-diff, user-directory friendly variant of D_discretization.py.

Original behavior:
- Reads:  Data/genie_normalized_1216.csv
- Keeps only specific GENIE assertion statuses and maps them to a controlled vocabulary
- Filters CUIs that have at least K occurrences in any single status category
- Writes: Data/genie_discretized_1216.csv

Key changes vs original:
- Removes hard-coded base directory + os.chdir()
- Adds CLI args for paths and K threshold (relative paths resolved against --base-dir)
"""
import argparse
import time
from pathlib import Path
import pandas as pd

def resolve_path(p: str, base_dir: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return Path(base_dir) / pp

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=".", help="Base directory for relative paths (default: current dir).")
    ap.add_argument("--input-csv", default="Data/genie_normalized_1216.csv",
                    help="Input CSV from C step (must contain umls_top_cui and assertion_status).")
    ap.add_argument("--output-csv", default="Data/genie_discretized_1216.csv",
                    help="Output CSV (patient_id, umls_top_cui, assertion_status_norm).")
    ap.add_argument("--K", type=int, default=10, help="Minimum count threshold per CUI across statuses.")
    args = ap.parse_args()

    input_path = resolve_path(args.input_csv, args.base_dir)
    output_path = resolve_path(args.output_csv, args.base_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    df = pd.read_csv(input_path)

    status_map = {
        "present": "PRESENT",
        "absent": "ABSENT",
        "historical": "HISTORY",
        "associated_with_someone_else": "FAMILY_HISTORY",
        "possible": "UNCERTAIN",
        "hypothetical": "UNCERTAIN",
        "conditional": "UNCERTAIN",
    }

    df2 = df.copy()
    df2["assertion_status_norm"] = df2["assertion_status"].map(status_map)
    df2 = df2[df2["assertion_status_norm"].notna()]

    counts = (
        df2.groupby(["umls_top_cui", "assertion_status_norm"])
           .size()
           .unstack(fill_value=0)
    )

    keep_cuis = counts.index[counts.max(axis=1) >= args.K]
    df_filtered = df2[df2["umls_top_cui"].isin(keep_cuis)].copy()

    df_discretized = df_filtered[["patient_id", "umls_top_cui", "assertion_status_norm"]].reset_index(drop=True)

    df_discretized.to_csv(output_path, index=False)
    print(f"Wrote {len(df_discretized)} rows to {output_path}")

    t1 = time.perf_counter()
    print(f"Total elapsed: {t1 - t0:.3f} seconds")

if __name__ == "__main__":
    main()
