#!/usr/bin/env python3
"""
C_normalization_portable.py

Minimal-diff, user-directory friendly variant of C_normalization.py.

Original behavior:
- Positional (row-by-row) alignment between:
    Data/genie_processed.csv  and  Data/vector_normalized.csv
- Produces:
    Data/genie_normalized_1216.csv

Key changes vs original:
- Removes hard-coded base directory + os.chdir()
- Adds CLI args for paths (relative paths resolved against --base-dir)
- Adds optional id-based alignment (still defaults to positional)

Notes:
- The external normalization output (vector_normalized.csv) often preserves the original row order.
  Default positional mode matches your original script.
"""
import argparse
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

    ap.add_argument("--input-norm", default="Data/vector_normalized.csv",
                    help="Normalization output CSV (must contain umls_top_cui).")
    ap.add_argument("--input-full", default="Data/genie_processed.csv",
                    help="Full data CSV from A step.")
    ap.add_argument("--output-csv", default="Data/genie_normalized_1216.csv",
                    help="Output normalized CSV.")

    ap.add_argument("--align", choices=["positional", "id"], default="positional",
                    help="Alignment mode. 'positional' matches original behavior.")
    ap.add_argument("--id-col", default="id",
                    help="ID column name for --align id. In id mode, this script will generate an id column "
                         "for the full data as 1..N if missing.")
    args = ap.parse_args()

    input_norm = resolve_path(args.input_norm, args.base_dir)
    input_full = resolve_path(args.input_full, args.base_dir)
    output_path = resolve_path(args.output_csv, args.base_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_full)
    normalize = pd.read_csv(input_norm)

    if "umls_top_cui" not in normalize.columns:
        raise ValueError("Normalization CSV missing required column: umls_top_cui")

    if args.align == "positional":
        df_selected = pd.concat(
            [
                df[["patient_id"]].reset_index(drop=True),
                normalize[["umls_top_cui"]].reset_index(drop=True),
                df[["semantic_type", "assertion_status", "value", "unit"]].reset_index(drop=True),
            ],
            axis=1,
        )
    else:
        # id-based alignment (optional)
        id_col = args.id_col
        if id_col not in normalize.columns:
            raise ValueError(f"--align id requires '{id_col}' in normalization CSV.")

        df2 = df.copy()
        if id_col not in df2.columns:
            df2[id_col] = range(1, len(df2) + 1)

        # Keep only the needed columns; join by id.
        merged = df2[[id_col, "patient_id", "semantic_type", "assertion_status", "value", "unit"]].merge(
            normalize[[id_col, "umls_top_cui"]],
            on=id_col,
            how="inner",
            validate="one_to_one",
        )
        df_selected = merged[["patient_id", "umls_top_cui", "semantic_type", "assertion_status", "value", "unit"]]

    df_selected.to_csv(output_path, index=False)
    print(f"Wrote {len(df_selected)} rows to {output_path}")

if __name__ == "__main__":
    main()
