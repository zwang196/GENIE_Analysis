#!/usr/bin/env python3
"""
TopK_CUI_Pairs.py

Find top cosine-similar CUI pairs per ordered (statusA ~ statusB) configuration.

Supports two input formats:
  A) pair_token + embeddings:
        pair_token = "C0000618||PRESENT"   (delim configurable)
        emb_0, emb_1, ...

  B) CUI column + embeddings (status implicit or separate):
        umls_top_cui (or other --cui_col)
        [optional status column via --status_col]
        emb_0, emb_1, ...
    If no status provided, all rows get --default_status (default: PRESENT).

Outputs:
  a_cui, a_assertion_status, b_cui, b_assertion_status, cosine_similarity, config
"""

import argparse
import heapq
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


def split_pair_token(x: str, delim: str = "||") -> Tuple[str, str]:
    """Split 'C0000618||HISTORY' -> ('C0000618', 'HISTORY')."""
    s = str(x)
    if delim in s:
        a, b = s.split(delim, 1)
        return a.strip(), b.strip()
    if "|" in s:  # fallback
        a, b = s.split("|", 1)
        return a.strip(), b.strip()
    return s.strip(), ""


def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize; float32; safe for NaN/Inf."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return (X / norms).astype(np.float32, copy=False)


def push_topk(heap: List[Tuple[float, int, int]], item: Tuple[float, int, int], k: int) -> None:
    """Maintain a min-heap of fixed size k for (sim, i_global, j_global)."""
    if k <= 0:
        return
    if len(heap) < k:
        heapq.heappush(heap, item)
    else:
        if item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)


def read_table(path: str, sep: Optional[str], auto_sep: bool) -> pd.DataFrame:
    if auto_sep:
        # pandas will sniff delimiter; can be slower but robust for CSV/TSV.
        return pd.read_csv(path, sep=None, engine="python")
    return pd.read_csv(path, sep=sep)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)

    # Input parsing
    ap.add_argument("--pair_token_col", default="pair_token",
                    help="Column like 'C0000618||PRESENT'. If missing, uses --cui_col/--status_col.")
    ap.add_argument("--cui_col", default="umls_top_cui",
                    help="CUI column for the 'all-present' (or separate status) format.")
    ap.add_argument("--status_col", default="",
                    help="Optional status column. If empty or missing, uses --default_status.")
    ap.add_argument("--default_status", default="PRESENT",
                    help="Status to assign when status is not provided.")

    ap.add_argument("--emb_prefix", default="emb_")
    ap.add_argument("--delim", default="||")

    # File delimiter handling
    ap.add_argument("--sep", default=",", help="CSV separator (use '\\t' for TSV). Ignored if --auto_sep.")
    ap.add_argument("--auto_sep", action="store_true",
                    help="Auto-detect separator (useful if you may have CSV or TSV).")

    # KNN / retention policy
    ap.add_argument("--k_search", type=int, default=2000,
                    help="Neighbors per query for EACH ordered configuration search.")
    ap.add_argument("--top_pp", type=int, default=100,
                    help="How many pairs to keep for pp_label~pp_label.")
    ap.add_argument("--top_other", type=int, default=10,
                    help="How many pairs to keep for all other ordered configs.")
    ap.add_argument("--pp_label", default="PRESENT",
                    help="Status label used for the pp_label~pp_label bucket (default: PRESENT).")

    ap.add_argument("--require_faiss", action="store_true",
                    help="If set, error out when FAISS is not available (otherwise fall back to brute force).")
    ap.add_argument("--faiss-gpu", action="store_true",
                    help="Use FAISS GPU index (requires faiss-gpu and CUDA).")

    args = ap.parse_args()

    # Read
    df = read_table(args.input_csv, sep=args.sep, auto_sep=args.auto_sep)

    # Embeddings
    emb_cols = [c for c in df.columns if c.startswith(args.emb_prefix)]
    if not emb_cols:
        raise ValueError(f"No embedding columns found with prefix '{args.emb_prefix}'.")

    # Identify cui/status
    has_pair_token = args.pair_token_col in df.columns
    has_cui_col = args.cui_col in df.columns

    if not has_pair_token and not has_cui_col:
        raise ValueError(
            f"Input must contain either '{args.pair_token_col}' OR '{args.cui_col}'. "
            f"Columns seen: {list(df.columns)[:20]} ..."
        )

    if has_pair_token:
        parsed = df[args.pair_token_col].apply(lambda x: split_pair_token(x, args.delim))
        cuis = np.array([p[0] for p in parsed], dtype=object)
        statuses = np.array([p[1] for p in parsed], dtype=object)
        # If status missing in some rows, fill with default
        statuses = np.where((statuses == "") | pd.isna(statuses), args.default_status, statuses).astype(object)
    else:
        cuis = df[args.cui_col].astype(str).to_numpy(dtype=object)
        if args.status_col and args.status_col in df.columns:
            statuses = df[args.status_col].astype(str).to_numpy(dtype=object)
            statuses = np.where((statuses == "") | pd.isna(statuses), args.default_status, statuses).astype(object)
        else:
            statuses = np.full(shape=(len(df),), fill_value=args.default_status, dtype=object)

    # Normalize embeddings
    X = l2_normalize_rows(df[emb_cols].to_numpy(dtype=np.float32))
    n, d = X.shape

    # Status grouping
    unique_statuses = sorted(set(statuses.tolist()))
    status_to_global_idx: Dict[str, np.ndarray] = {
        s: np.where(statuses == s)[0].astype(np.int64) for s in unique_statuses
    }
    status_to_X: Dict[str, np.ndarray] = {s: X[idx] for s, idx in status_to_global_idx.items()}

    def limit_for(cfg: str) -> int:
        return args.top_pp if cfg == f"{args.pp_label}~{args.pp_label}" else args.top_other

    # Try FAISS
    use_faiss = True
    try:
        import faiss  # type: ignore
    except Exception:
        use_faiss = False
        if args.require_faiss:
            raise RuntimeError(
                "FAISS is required (because --require_faiss was set), but it could not be imported.\n"
                "Install: pip install faiss-cpu (or faiss-gpu)"
            )

    heaps: Dict[str, List[Tuple[float, int, int]]] = defaultdict(list)

    if use_faiss:
        # Build one FAISS index per B-status
        status_to_index: Dict[str, "faiss.Index"] = {}
        use_gpu = bool(args.faiss_gpu)
        gpu_res = None
        if use_gpu:
            if not hasattr(faiss, "get_num_gpus"):
                raise RuntimeError("FAISS GPU support not available; install faiss-gpu.")
            if faiss.get_num_gpus() <= 0:
                raise RuntimeError("FAISS GPU requested but no GPU is visible.")
            gpu_res = faiss.StandardGpuResources()

        for sB in unique_statuses:
            xb = status_to_X[sB]
            index_cpu = faiss.IndexFlatIP(d)  # cosine via IP on normalized vectors
            index_cpu.add(xb)
            if use_gpu:
                index = faiss.index_cpu_to_gpu(gpu_res, 0, index_cpu)
            else:
                index = index_cpu
            status_to_index[sB] = index

        for sA in unique_statuses:
            idxA_global = status_to_global_idx[sA]
            XA = status_to_X[sA]
            if idxA_global.size == 0:
                continue

            for sB in unique_statuses:
                idxB_global = status_to_global_idx[sB]
                if idxB_global.size == 0:
                    continue

                cfg = f"{sA}~{sB}"
                k_keep = limit_for(cfg)

                indexB = status_to_index[sB]
                # If sA==sB, include self in KNN and filter it out
                k = min(args.k_search + (1 if sA == sB else 0), idxB_global.size)

                sims, nbrs = indexB.search(XA, k)  # local-to-B indices

                if sA == sB:
                    for q_local in range(XA.shape[0]):
                        i_global = int(idxA_global[q_local])
                        for sim, j_local in zip(sims[q_local], nbrs[q_local]):
                            j_local = int(j_local)
                            if j_local == q_local:
                                continue
                            j_global = int(idxB_global[j_local])
                            push_topk(heaps[cfg], (float(sim), i_global, j_global), k_keep)
                else:
                    for q_local in range(XA.shape[0]):
                        i_global = int(idxA_global[q_local])
                        for sim, j_local in zip(sims[q_local], nbrs[q_local]):
                            j_global = int(idxB_global[int(j_local)])
                            push_topk(heaps[cfg], (float(sim), i_global, j_global), k_keep)

    else:
        # Brute-force fallback (O(n^2) per status-pair). Safe for small datasets.
        # Uses matrix multiplication on normalized vectors (cosine = dot product).
        for sA in unique_statuses:
            idxA_global = status_to_global_idx[sA]
            XA = status_to_X[sA]
            if idxA_global.size == 0:
                continue

            for sB in unique_statuses:
                idxB_global = status_to_global_idx[sB]
                XB = status_to_X[sB]
                if idxB_global.size == 0:
                    continue

                cfg = f"{sA}~{sB}"
                k_keep = limit_for(cfg)

                # Compute similarities (|A| x |B|)
                S = XA @ XB.T  # cosine on normalized vectors
                if sA == sB:
                    # remove self matches on diagonal
                    m = min(S.shape[0], S.shape[1])
                    S[np.arange(m), np.arange(m)] = -np.inf

                # For each row, take top args.k_search candidates, then push into heap
                k_row = min(args.k_search, S.shape[1])
                if k_row <= 0:
                    continue

                # partial sort per row
                for q_local in range(S.shape[0]):
                    i_global = int(idxA_global[q_local])
                    row = S[q_local]
                    if k_row == row.shape[0]:
                        cand_idx = np.arange(row.shape[0])
                    else:
                        cand_idx = np.argpartition(row, -k_row)[-k_row:]
                    # sort those candidates descending
                    cand_idx = cand_idx[np.argsort(row[cand_idx])[::-1]]
                    for j_local in cand_idx:
                        sim = float(row[int(j_local)])
                        if not np.isfinite(sim):
                            continue
                        j_global = int(idxB_global[int(j_local)])
                        push_topk(heaps[cfg], (sim, i_global, j_global), k_keep)

    # Build output
    rows = []
    for cfg, h in heaps.items():
        for sim, i, j in sorted(h, key=lambda x: x[0], reverse=True):
            rows.append({
                "a_cui": cuis[i],
                "a_assertion_status": statuses[i],
                "b_cui": cuis[j],
                "b_assertion_status": statuses[j],
                "cosine_similarity": float(sim),
                "config": cfg,
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["a_assertion_status", "b_assertion_status", "cosine_similarity"],
            ascending=[True, True, False],
            kind="mergesort",
        )

    out.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(out)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
