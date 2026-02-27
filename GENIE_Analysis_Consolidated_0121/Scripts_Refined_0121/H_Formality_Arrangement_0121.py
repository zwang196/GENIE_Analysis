#!/usr/bin/env python3
"""
H_Formality_Arrangement_portable.py

Minimal-diff, user-directory friendly variant of H_Formality_Arrangement.py.

Key changes vs original:
- Adds CLI wrapper (instead of running two hard-coded example calls)
- Keeps the same enrichment logic: join pairs CSV with codebook (CUI -> term/semantic_type)

Example:
  python H_Formality_Arrangement_portable.py \
    --pairs-csv Output/baseline_1216_positive_pairs.csv \
    --codebook-csv Data/vector_normalized.csv \
    --out-csv Output/baseline_1216_pairs_explained.csv
"""
import argparse
from pathlib import Path
import pandas as pd

def resolve_path(p: str, base_dir: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return Path(base_dir) / pp

def build_cui_map(codebook_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce codebook to a unique CUI->(term, semantic_type) mapping.
    If multiple rows share the same CUI, keep the one with highest umls_top_confidence (if present),
    otherwise keep the first occurrence.
    """
    need = {"umls_top_cui", "umls_top_term", "umls_top_semantic_type"}
    missing = need - set(codebook_df.columns)
    if missing:
        raise ValueError(f"Codebook missing required columns: {sorted(missing)}")

    cb = codebook_df.copy()
    cb["umls_top_cui"] = cb["umls_top_cui"].astype(str).str.strip()

    if "umls_top_confidence" in cb.columns:
        cb["umls_top_confidence"] = pd.to_numeric(cb["umls_top_confidence"], errors="coerce").fillna(-1.0)
        cb = cb.sort_values("umls_top_confidence", ascending=False)

    cb = cb.drop_duplicates(subset=["umls_top_cui"], keep="first")

    return cb[["umls_top_cui", "umls_top_term", "umls_top_semantic_type"]].rename(columns={
        "umls_top_cui": "cui",
        "umls_top_term": "term",
        "umls_top_semantic_type": "semantic_type",
    })

def enrich_pairs(pairs_csv: str, codebook_csv: str, out_csv: str) -> None:
    pairs = pd.read_csv(pairs_csv)
    codebook = pd.read_csv(codebook_csv)

    required_pairs = {"a_cui", "a_assertion_status", "b_cui", "b_assertion_status"}
    missing = required_pairs - set(pairs.columns)
    if missing:
        raise ValueError(f"Pairs file missing required columns: {sorted(missing)}")

    cmap = build_cui_map(codebook)

    pairs["a_cui"] = pairs["a_cui"].astype(str).str.strip()
    pairs["b_cui"] = pairs["b_cui"].astype(str).str.strip()

    out = pairs.merge(
        cmap.rename(columns={"cui": "a_cui", "term": "a_term", "semantic_type": "a_semantic_type"}),
        on="a_cui",
        how="left",
    )

    out = out.merge(
        cmap.rename(columns={"cui": "b_cui", "term": "b_term", "semantic_type": "b_semantic_type"}),
        on="b_cui",
        how="left",
    )

    out.insert(0, "pair_id", range(1, len(out) + 1))

    out = out[[
        "pair_id",
        "a_cui", "a_term", "a_semantic_type", "a_assertion_status",
        "b_cui", "b_term", "b_semantic_type", "b_assertion_status",
    ]]

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Wrote {len(out)} rows to: {out_csv}")

    unmapped_a = out["a_term"].isna().sum()
    unmapped_b = out["b_term"].isna().sum()
    if unmapped_a or unmapped_b:
        print(f"Warning: unmapped CUIs -> a_side: {unmapped_a}, b_side: {unmapped_b}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=".", help="Base directory for relative paths (default: current dir).")
    ap.add_argument("--pairs-csv", required=True, help="Pairs CSV (e.g., output of E/G).")
    ap.add_argument("--codebook-csv", required=True, help="Codebook CSV with umls_top_* columns.")
    ap.add_argument("--out-csv", required=True, help="Output enriched pairs CSV.")
    args = ap.parse_args()

    pairs_csv = resolve_path(args.pairs_csv, args.base_dir)
    codebook_csv = resolve_path(args.codebook_csv, args.base_dir)
    out_csv = resolve_path(args.out_csv, args.base_dir)

    enrich_pairs(str(pairs_csv), str(codebook_csv), str(out_csv))

if __name__ == "__main__":
    main()
