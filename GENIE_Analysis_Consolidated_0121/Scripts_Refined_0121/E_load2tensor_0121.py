#!/usr/bin/env python3
"""
E_load2tensor_0121.py

Build a sparse SPPMI matrix from discretized EHR data.

Input (from Step 4):
  - CSV with columns: patient_id, umls_top_cui, assertion_status_norm

Typical usage:
  python E_load2tensor_0121.py \
    --input_csv Data/genie_discretized.csv \
    --output_npz Data/genie_discretized_sppmi.npz \
    --vocab_csv Data/genie_discretized_cui_vocab.csv \
    --status_filter ALL \
    --min_cui_count 1 --min_cooccur 1

Notes:
  - Builds tokens as "CUI||STATUS" (delimiter configurable) to preserve assertion status.
  - Use --present_only to restrict to PRESENT only.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp


def parse_status_filter(s: str) -> Optional[Set[str]]:
    if s is None:
        return None
    ss = s.strip()
    if ss == "" or ss.upper() == "ALL":
        return None
    return {x.strip().upper() for x in ss.split(",") if x.strip()}


def iter_patient_sets(
    df: pd.DataFrame,
    patient_col: str,
    token_col: str,
    max_tokens_per_patient: int,
) -> Iterable[Sequence[str]]:
    for _, sub in df.groupby(patient_col, sort=False):
        tokens = pd.unique(sub[token_col].astype(str))
        if max_tokens_per_patient > 0 and len(tokens) > max_tokens_per_patient:
            tokens = np.sort(tokens)[:max_tokens_per_patient]
        yield tokens.tolist()


def split_token(token: str, delim: str) -> Tuple[str, str]:
    if delim and delim in token:
        a, b = token.split(delim, 1)
        return a, b
    return token, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True, help="Discretized CSV from Step 4.")
    ap.add_argument("--output_npz", required=True, help="Output SPPMI matrix (.npz).")
    ap.add_argument("--vocab_csv", default="", help="Output CUI vocab CSV (optional).")

    ap.add_argument("--patient_col", default="patient_id")
    ap.add_argument("--cui_col", default="umls_top_cui")
    ap.add_argument("--status_col", default="assertion_status_norm")
    ap.add_argument("--status_filter", default="ALL",
                    help="Comma-separated status filter (e.g., PRESENT,HISTORY). Use ALL for no filter.")
    ap.add_argument("--present_only", action="store_true",
                    help="Shortcut: only include PRESENT status (overrides status_filter).")
    ap.add_argument("--token_delim", default="||",
                    help="Delimiter for CUI~STATUS tokens (default: '||').")

    ap.add_argument("--min_cui_count", type=int, default=1,
                    help="Minimum patient count per token to keep.")
    ap.add_argument("--min_cooccur", type=int, default=1,
                    help="Minimum co-occurrence count for a pair to keep.")
    ap.add_argument("--max_cuis_per_patient", type=int, default=0,
                    help="Cap CUIs per patient (0 = no cap).")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--device", default="cpu",
                    help="cpu or cuda (cuda requires cupy).")

    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    for col in (args.patient_col, args.cui_col, args.status_col):
        if col not in df.columns:
            raise SystemExit(f"Missing required column: {col}")

    # Optional status filter
    status_filter = args.status_filter
    if args.present_only:
        if status_filter and status_filter.strip().upper() not in {"", "ALL", "PRESENT"}:
            print("[WARN] present_only set; overriding status_filter to PRESENT.")
        status_filter = "PRESENT"
    filt = parse_status_filter(status_filter)

    df = df[[args.patient_col, args.cui_col, args.status_col]].dropna()
    df[args.cui_col] = df[args.cui_col].astype(str).str.strip()
    df[args.status_col] = df[args.status_col].astype(str).str.upper().str.strip()
    if filt is not None:
        df = df[df[args.status_col].isin(filt)]

    if df.empty:
        raise SystemExit("No rows remaining after filtering; cannot build SPPMI.")

    token_col = "__cui_status_token"
    df[token_col] = df[args.cui_col] + args.token_delim + df[args.status_col]

    # First pass: count tokens by patient occurrence
    token_patient_counts: Dict[str, int] = defaultdict(int)
    patient_sets: List[List[str]] = []
    for tokens in iter_patient_sets(df, args.patient_col, token_col, args.max_cuis_per_patient):
        if not tokens:
            continue
        patient_sets.append(tokens)
        for t in set(tokens):
            token_patient_counts[t] += 1

    # Filter tokens by minimum count
    keep_tokens = {t for t, n in token_patient_counts.items() if n >= args.min_cui_count}
    if not keep_tokens:
        raise SystemExit("No tokens meet min_cui_count; nothing to build.")

    vocab = sorted(keep_tokens)
    token_to_idx = {t: i for i, t in enumerate(vocab)}
    print(f"[INFO] Tokens kept: {len(vocab):,}")

    # Build co-occurrence counts (patient-level binary co-occurrence)
    cooc: Dict[Tuple[int, int], int] = defaultdict(int)
    for tokens in patient_sets:
        idxs = [token_to_idx[t] for t in set(tokens) if t in token_to_idx]
        if len(idxs) < 2:
            continue
        idxs = sorted(idxs)
        if args.max_cuis_per_patient > 0 and len(idxs) > args.max_cuis_per_patient:
            idxs = idxs[:args.max_cuis_per_patient]
        for i_pos in range(len(idxs)):
            i = idxs[i_pos]
            for j in idxs[i_pos + 1:]:
                cooc[(i, j)] += 1

    if not cooc:
        raise SystemExit("No co-occurrences found; cannot build SPPMI.")

    # Expand to symmetric matrix and apply min_cooccur
    rows: List[int] = []
    cols: List[int] = []
    data: List[int] = []
    for (i, j), v in cooc.items():
        if v < args.min_cooccur:
            continue
        rows.append(i); cols.append(j); data.append(v)
        rows.append(j); cols.append(i); data.append(v)

    if not data:
        raise SystemExit("All pairs filtered by min_cooccur; cannot build SPPMI.")

    n = len(vocab)
    mat = sp.coo_matrix((np.array(data, dtype=np.float64), (rows, cols)), shape=(n, n)).tocsr()
    mat.sum_duplicates()

    # Compute PPMI
    row_sums = np.asarray(mat.sum(axis=1)).ravel()
    col_sums = np.asarray(mat.sum(axis=0)).ravel()
    total = float(mat.sum())
    if total <= 0:
        raise SystemExit("Total co-occurrence is zero; cannot compute PMI.")

    mat_coo = mat.tocoo()
    denom = row_sums[mat_coo.row] * col_sums[mat_coo.col]

    if args.device.lower().startswith("cuda"):
        try:
            import cupy as cp  # type: ignore
        except Exception as e:
            raise SystemExit(f"device=cuda requires cupy. Import error: {e}")

        data_gpu = cp.asarray(mat_coo.data)
        denom_gpu = cp.asarray(denom)
        total_gpu = cp.asarray(total, dtype=data_gpu.dtype)

        # Avoid version-specific cp.errstate support by computing PMI with
        # explicit masking. This works across CuPy variants.
        numer = data_gpu * total_gpu
        ratio = cp.zeros_like(data_gpu)
        valid_denom = denom_gpu != 0
        ratio[valid_denom] = numer[valid_denom] / denom_gpu[valid_denom]

        pmi = cp.full_like(ratio, -cp.inf)
        valid_ratio = ratio > 0
        pmi[valid_ratio] = cp.log(ratio[valid_ratio])
        ppmi = cp.maximum(pmi, 0.0)
        ppmi = cp.asnumpy(ppmi)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log((mat_coo.data * total) / denom)
        ppmi = np.maximum(pmi, 0.0)

    dtype = np.float32 if args.dtype == "float32" else np.float64
    ppmi = ppmi.astype(dtype, copy=False)
    sppmi = sp.coo_matrix((ppmi, (mat_coo.row, mat_coo.col)), shape=mat.shape).tocsr()
    sppmi.eliminate_zeros()

    sp.save_npz(args.output_npz, sppmi)
    print(f"[OK] Saved SPPMI: {args.output_npz}")
    print(f"[INFO] Shape: {sppmi.shape} nnz={sppmi.nnz:,}")

    # Save vocab
    vocab_path = args.vocab_csv.strip()
    if not vocab_path:
        vocab_path = args.output_npz + ".vocab.csv"
    cuis: List[str] = []
    statuses: List[str] = []
    for tok in vocab:
        c, s = split_token(tok, args.token_delim)
        cuis.append(c)
        statuses.append(s)

    out_df = pd.DataFrame({
        "idx": np.arange(len(vocab), dtype=int),
        "token": vocab,
        "cui": cuis,
        "status": statuses,
        "patient_count": [token_patient_counts[t] for t in vocab],
    })
    out_df.to_csv(vocab_path, index=False)
    print(f"[OK] Saved vocab: {vocab_path}")


if __name__ == "__main__":
    main()
