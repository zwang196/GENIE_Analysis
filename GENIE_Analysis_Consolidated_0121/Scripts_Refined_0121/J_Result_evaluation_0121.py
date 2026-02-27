#!/usr/bin/env python3
"""
Apply hard cutoffs to relationship_score to produce deterministic labels.

Adds columns:
  - has_medical_relationship_cutoff (TRUE/FALSE or NA if tri-state and uncertain)
  - relationship_label_cutoff ("positive"/"negative"/"uncertain" if tri-state)

Also prints summary stats.

Usage examples:
  # Binary cutoff at 70
  python apply_score_cutoffs.py --input out.tsv --output out.labeled.tsv --threshold 70

  # Tri-state: <=50 negative, >=85 positive, otherwise uncertain
  python apply_score_cutoffs.py --input out.csv --output out.labeled.csv --low 50 --high 85

  # Chunked processing for huge files
  python apply_score_cutoffs.py --input out.csv --output out.labeled.csv --low 50 --high 85 --chunksize 200000
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple

import pandas as pd


def detect_sep(path: str, sep: Optional[str]) -> str:
    if sep is not None:
        return sep
    ext = os.path.splitext(path)[1].lower()
    if ext in {".tsv", ".tab"}:
        return "\t"
    return ","


def ensure_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def apply_cutoffs(
    df: pd.DataFrame,
    score_col: str,
    mode: str,
    threshold: Optional[float],
    low: Optional[float],
    high: Optional[float],
) -> pd.DataFrame:
    score = ensure_numeric(df[score_col])

    if mode == "binary":
        if threshold is None:
            raise ValueError("Binary mode requires --threshold.")
        label = score >= float(threshold)
        df["has_medical_relationship_cutoff"] = label.map({True: "TRUE", False: "FALSE"})
        df["relationship_label_cutoff"] = label.map({True: "positive", False: "negative"})

    elif mode == "tristate":
        if low is None or high is None:
            raise ValueError("Tri-state mode requires --low and --high.")
        low_v, high_v = float(low), float(high)
        if low_v >= high_v:
            raise ValueError(f"Expected low < high, got low={low_v}, high={high_v}.")

        # Default: uncertain
        label_str = pd.Series(["uncertain"] * len(df), index=df.index, dtype="string")
        label_str = label_str.mask(score <= low_v, "negative")
        label_str = label_str.mask(score >= high_v, "positive")
        df["relationship_label_cutoff"] = label_str

        # For convenience, provide a boolean-like column:
        # - TRUE for positive
        # - FALSE for negative
        # - empty for uncertain or missing score
        bool_like = pd.Series([""] * len(df), index=df.index, dtype="string")
        bool_like = bool_like.mask(df["relationship_label_cutoff"] == "positive", "TRUE")
        bool_like = bool_like.mask(df["relationship_label_cutoff"] == "negative", "FALSE")
        # If score is NaN, keep empty
        df["has_medical_relationship_cutoff"] = bool_like

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return df


def summarize(
    df: pd.DataFrame,
    score_col: str,
    conf_col: str,
    label_col: str,
) -> None:
    score = ensure_numeric(df[score_col])
    conf = ensure_numeric(df[conf_col]) if conf_col in df.columns else pd.Series(dtype="float64")

    def num_desc(x: pd.Series) -> dict:
        x = x.dropna()
        return {
            "count": int(x.count()),
            "mean": float(x.mean()) if len(x) else float("nan"),
            "std": float(x.std(ddof=1)) if len(x) > 1 else float("nan"),
            "min": float(x.min()) if len(x) else float("nan"),
            "p25": float(x.quantile(0.25)) if len(x) else float("nan"),
            "median": float(x.quantile(0.50)) if len(x) else float("nan"),
            "p75": float(x.quantile(0.75)) if len(x) else float("nan"),
            "max": float(x.max()) if len(x) else float("nan"),
        }

    print("\n=== Numeric Summary ===")
    score_desc = num_desc(score)
    print(f"{score_col}: {score_desc}")
    if conf_col in df.columns:
        conf_desc = num_desc(conf)
        print(f"{conf_col}:  {conf_desc}")
    else:
        print(f"{conf_col}:  (column not present)")

    print("\n=== Derived Label Distribution ===")
    if label_col not in df.columns:
        print(f"{label_col}: (column not present)")
        return
    vc = df[label_col].astype("string").fillna("").value_counts(dropna=False)
    total = int(vc.sum())
    for k, v in vc.items():
        key = k if k != "" else "<empty>"
        print(f"{key}: {int(v)} ({(int(v)/total)*100:.2f}%)")


def process_file(
    in_path: str,
    out_path: str,
    sep: Optional[str],
    score_col: str,
    conf_col: str,
    mode: str,
    threshold: Optional[float],
    low: Optional[float],
    high: Optional[float],
    chunksize: Optional[int],
) -> None:
    chosen_sep = detect_sep(in_path, sep)
    out_sep = detect_sep(out_path, sep)

    # We will stream if chunksize is provided; otherwise read fully.
    if chunksize:
        first = True
        summary_accum = []
        for chunk in pd.read_csv(in_path, sep=chosen_sep, dtype="string", chunksize=chunksize):
            if score_col not in chunk.columns:
                raise ValueError(f"Missing required column: {score_col}")
            chunk = apply_cutoffs(chunk, score_col, mode, threshold, low, high)

            # Write incrementally
            chunk.to_csv(out_path, sep=out_sep, index=False, mode="w" if first else "a", header=first)
            first = False

            # Keep a small sample for summary (or accumulate label counts)
            summary_accum.append(chunk[[score_col] + ([conf_col] if conf_col in chunk.columns else []) + ["relationship_label_cutoff"]].copy())

        # Summarize on concatenated sampled chunks (this is fine for descriptive stats if you want exact,
        # remove sampling and instead accumulate streaming stats; for most use cases this is acceptable).
        df_sum = pd.concat(summary_accum, ignore_index=True)
        summarize(df_sum, score_col=score_col, conf_col=conf_col, label_col="relationship_label_cutoff")

    else:
        df = pd.read_csv(in_path, sep=chosen_sep, dtype="string")
        if score_col not in df.columns:
            raise ValueError(f"Missing required column: {score_col}")
        df = apply_cutoffs(df, score_col, mode, threshold, low, high)
        df.to_csv(out_path, sep=out_sep, index=False)
        summarize(df, score_col=score_col, conf_col=conf_col, label_col="relationship_label_cutoff")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input CSV/TSV path")
    p.add_argument("--output", required=True, help="Output CSV/TSV path")
    p.add_argument("--sep", default=None, help="Delimiter override, e.g. ',' or '\\t'")

    p.add_argument("--score_col", default="relationship_score", help="Score column name")
    p.add_argument("--conf_col", default="confidence", help="Confidence column name (optional)")

    p.add_argument("--mode", choices=["binary", "tristate"], default="tristate")
    p.add_argument("--threshold", type=float, default=None, help="Binary cutoff: score >= threshold => positive")
    p.add_argument("--low", type=float, default=50.0, help="Tri-state low cutoff: score <= low => negative")
    p.add_argument("--high", type=float, default=85.0, help="Tri-state high cutoff: score >= high => positive")

    p.add_argument("--chunksize", type=int, default=None, help="Process in chunks (rows) for very large files")

    args = p.parse_args()

    try:
        process_file(
            in_path=args.input,
            out_path=args.output,
            sep=args.sep,
            score_col=args.score_col,
            conf_col=args.conf_col,
            mode=args.mode,
            threshold=args.threshold,
            low=args.low,
            high=args.high,
            chunksize=args.chunksize,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
