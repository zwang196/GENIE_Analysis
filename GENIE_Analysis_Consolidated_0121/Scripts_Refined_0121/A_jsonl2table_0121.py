#!/usr/bin/env python3
"""
A_jsonl2table_portable.py

Minimal-diff, user-directory friendly variant of A_jsonl2table.py.

Key changes vs original:
- Removes hard-coded base directory + os.chdir()
- Adds CLI args for input/output paths (relative paths resolved against --base-dir)
- Allows key delimiter to be configured

Example:
  python A_jsonl2table_portable.py \
    --base-dir /path/to/GENIE_Analysis \
    --input-jsonl Data/genie_train.jsonl \
    --output-csv Data/genie_processed.csv
"""
import argparse
import time
import json
import csv
from pathlib import Path

def resolve_path(p: str, base_dir: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return Path(base_dir) / pp

def parse_key(key_str, delim: str = "-"):
    if key_str is None:
        return None, None, None
    parts = str(key_str).split(delim)
    if len(parts) != 3:
        # Unexpected pattern – fall back to None for safety
        return None, None, None
    patient_id, admission_id, message_id = parts
    return patient_id, admission_id, message_id

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=".", help="Base directory for relative paths (default: current dir).")
    ap.add_argument("--input-jsonl", default="Data/genie_train.jsonl", help="Input GENIE .jsonl")
    ap.add_argument("--output-csv", default="Data/genie_processed.csv", help="Output .csv")
    ap.add_argument("--key-delim", default="-", help="Delimiter in GENIE key, e.g. '175-1-2' uses '-'")
    args = ap.parse_args()

    input_path = resolve_path(args.input_jsonl, args.base_dir)
    output_path = resolve_path(args.output_csv, args.base_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # Step 1.1: Read rows and collect keys
    rows = []
    item_keys = set()

    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                outer = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {line_no}: invalid JSON, skipping. Error: {e}")
                continue

            key = outer.get("key")

            # Parse nested "response" field: JSON-encoded string
            try:
                response_items = json.loads(outer["response"])
            except (KeyError, json.JSONDecodeError) as e:
                print(f"[WARN] Line {line_no}: cannot parse 'response', skipping. Error: {e}")
                continue

            for item in response_items:
                row = {"key": key}
                for k, v in item.items():
                    row[k] = v
                    item_keys.add(k)
                rows.append(row)

    # Step 1.2: Parse key -> patient_id/admission_id/message_id
    for r in rows:
        key = r.get("key")
        patient_id, admission_id, message_id = parse_key(key, delim=args.key_delim)
        r["patient_id"] = patient_id
        r["admission_id"] = admission_id
        r["message_id"] = message_id

    # Step 1.3: Columns (preserve original ordering)
    ordered_base = ["key", "patient_id", "admission_id", "message_id"]
    descript = ["phrase", "body_location", "semantic_type", "modifier", "purpose"]
    assertion = ["assertion_status", "value", "unit"]
    date = ["start_date", "end_date"]

    merged_keys = ordered_base + descript + assertion + date
    remaining_keys = sorted(k for k in item_keys if k not in merged_keys)
    fieldnames = merged_keys + remaining_keys

    # Step 1.4: Write CSV
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {field: row.get(field, None) for field in fieldnames}
            writer.writerow(out)

    print(f"Done. Wrote {len(rows)} rows to {output_path}")
    t1 = time.perf_counter()
    print(f"Total elapsed: {t1 - t0:.3f} seconds")

if __name__ == "__main__":
    main()
