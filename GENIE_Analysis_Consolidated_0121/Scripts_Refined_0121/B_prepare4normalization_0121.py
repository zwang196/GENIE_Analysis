#!/usr/bin/env python3
"""
B_prepare4normalization_portable.py

Minimal-diff, user-directory friendly variant of B_prepare4normalization.py.

Key changes vs original:
- Removes hard-coded base directory + os.chdir()
- Adds CLI args for input/output paths (relative paths resolved against --base-dir)

Example:
  python B_prepare4normalization_portable.py \
    --base-dir /path/to/GENIE_Analysis \
    --input-csv Data/genie_processed.csv \
    --output-csv Data/vector_for_normalization.csv
"""
import argparse
from pathlib import Path
import pandas as pd

def resolve_path(p: str, base_dir: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return Path(base_dir) / pp

def make_entity(row) -> str:
    phrase = row.get("phrase")
    body = row.get("body_location")

    phrase = str(phrase).strip() if pd.notna(phrase) else ""
    body = str(body).strip() if pd.notna(body) else ""

    if phrase and body:
        return f"{phrase} {body}"
    elif phrase:
        return phrase
    elif body:
        return body
    else:
        return ""

def make_note(row) -> str:
    parts = []
    for col in ["semantic_type", "modifier", "purpose"]:
        val = row.get(col)
        if pd.notna(val):
            s = str(val).strip()
            if s:
                parts.append(s)
    return ", ".join(parts)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=".", help="Base directory for relative paths (default: current dir).")
    ap.add_argument("--input-csv", default="Data/genie_processed.csv", help="Input .csv from A step")
    ap.add_argument("--output-csv", default="Data/vector_for_normalization.csv",
                    help="Output .csv for external UMLS normalization")
    args = ap.parse_args()

    input_path = resolve_path(args.input_csv, args.base_dir)
    output_path = resolve_path(args.output_csv, args.base_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    df_out = pd.DataFrame({
        "id": range(1, len(df) + 1),
        "entity": df.apply(make_entity, axis=1),
        "note": df.apply(make_note, axis=1),
    })

    df_out.to_csv(output_path, index=False)
    print(f"Wrote {len(df_out)} rows to {output_path}")

if __name__ == "__main__":
    main()
