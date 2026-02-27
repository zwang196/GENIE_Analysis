#!/usr/bin/env python3
"""
GPU-capable variant of F_2Ddecomp_1216.py.

CPU path (default):
  - SciPy sparse CSR/CSC -> scipy.sparse.linalg.eigsh (ARPACK)

GPU path:
  - SciPy sparse -> torch sparse COO on GPU -> torch.lobpcg (LOBPCG)

Also prints:
  - Wall-clock runtime
  - CPU usage (user/sys time, max RSS, estimated CPU utilization)
  - GPU usage (nvidia-smi snapshot + torch CUDA peak memory) when using CUDA

Notes:
  - For large sparse matrices, GPU path requires enough GPU memory to store the sparse tensor
    (nnz * (indices + values) can be substantial).
  - torch.lobpcg convergence can be more sensitive than eigsh; tune --lobpcg-*.

"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import resource
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def threshold_scipy_sparse(A: sp.spmatrix, thr: float) -> sp.csr_matrix:
    """Drop entries with value < thr. (Keeps >= thr)."""
    A = A.tocoo()
    keep = A.data >= thr
    if keep.all():
        return A.tocsr()
    A2 = sp.coo_matrix((A.data[keep], (A.row[keep], A.col[keep])), shape=A.shape, dtype=A.dtype)
    A2.sum_duplicates()
    return A2.tocsr()


def make_embeddings(evals: np.ndarray, evecs: np.ndarray, method: str) -> np.ndarray:
    """
    Convert eigendecomposition to embeddings.
    - 'evecs': raw eigenvectors
    - 'sqrt_evals': evecs * sqrt(max(evals,0))
    """
    if method == "evecs":
        return evecs
    if method == "sqrt_evals":
        w = np.sqrt(np.clip(evals, 0.0, None))
        return evecs * w.reshape(1, -1)
    raise ValueError(f"Unknown embedding method: {method}")


def _fmt_mb(x_bytes: float) -> str:
    return f"{x_bytes / (1024**2):,.2f} MiB"


def _cpu_usage_snapshot() -> Dict[str, Any]:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is KB on Linux
    maxrss_kb = float(getattr(ru, "ru_maxrss", 0.0))
    return {
        "user_s": float(getattr(ru, "ru_utime", 0.0)),
        "sys_s": float(getattr(ru, "ru_stime", 0.0)),
        "maxrss_mb": maxrss_kb / 1024.0,
    }


def _nvidia_smi_snapshot(gpu_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Snapshot GPU utilization & memory. Returns None if nvidia-smi is unavailable.
    """
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, text=True).strip().splitlines()
        rows = []
        for line in out:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 6:
                continue
            idx = int(parts[0])
            if gpu_id is not None and idx != gpu_id:
                continue
            rows.append({
                "gpu_index": idx,
                "name": parts[1],
                "util_gpu_pct": float(parts[2]),
                "util_mem_pct": float(parts[3]),
                "mem_used_mb": float(parts[4]),
                "mem_total_mb": float(parts[5]),
            })
        return {"gpus": rows}
    except Exception:
        return None


def _infer_gpu_id_from_device(device_str: str) -> Optional[int]:
    # Accept: "cuda", "cuda:0", "cuda:1"
    if not device_str.startswith("cuda"):
        return None
    if ":" in device_str:
        try:
            return int(device_str.split(":")[1])
        except Exception:
            return None
    return 0


def _estimate_cpu_util_pct(cpu_user_sys: float, wall_s: float, cores: int) -> float:
    if wall_s <= 0:
        return 0.0
    denom = max(1, cores) * wall_s
    return 100.0 * cpu_user_sys / denom


def scipy_to_torch_sparse_coo(A: sp.spmatrix, device: "torch.device", dtype: "torch.dtype"):
    """
    Convert SciPy sparse matrix to torch sparse COO tensor on target device.
    """
    import torch  # local import for CPU-only environments

    Acoo = A.tocoo()
    row = torch.from_numpy(Acoo.row.astype(np.int64, copy=False))
    col = torch.from_numpy(Acoo.col.astype(np.int64, copy=False))
    idx = torch.stack([row, col], dim=0)
    val = torch.from_numpy(Acoo.data.astype(np.float32 if dtype == torch.float32 else np.float64, copy=False))
    T = torch.sparse_coo_tensor(idx, val, size=Acoo.shape, device=device, dtype=dtype)
    return T.coalesce()


def torch_lobpcg_topk(
    A_torch, k: int, largest: bool, tol: float, niter: int, seed: int, init: str
) -> Tuple["torch.Tensor", "torch.Tensor", str]:
    """
    Compute top-k eigenpairs using torch.lobpcg given a sparse torch tensor A.
    Returns (evals, evecs) on the same device as A_torch.
    """
    import torch

    n = A_torch.size(0)
    g = torch.Generator(device=A_torch.device)
    g.manual_seed(seed)

    if init == "randn":
        X = torch.randn((n, k), device=A_torch.device, dtype=A_torch.dtype, generator=g)
    elif init == "rand":
        X = torch.rand((n, k), device=A_torch.device, dtype=A_torch.dtype, generator=g)
    else:
        raise ValueError(f"Unknown init: {init}")

    with torch.no_grad():
        # Prefer tensor input (works in newer torch), then fallback to callable
        # for builds that only support operator-like inputs.
        try:
            evals, evecs = torch.lobpcg(
                A=A_torch,
                k=k,
                X=X,
                niter=niter,
                tol=tol,
                largest=largest,
            )
            backend = "tensor"
        except Exception as e_tensor:
            def matmul(x: "torch.Tensor") -> "torch.Tensor":
                if A_torch.is_sparse:
                    return torch.sparse.mm(A_torch, x)
                return A_torch @ x

            try:
                evals, evecs = torch.lobpcg(
                    A=matmul,
                    k=k,
                    X=X,
                    niter=niter,
                    tol=tol,
                    largest=largest,
                )
                backend = "callable"
            except Exception as e_callable:
                raise RuntimeError(
                    "torch.lobpcg failed for both tensor and callable inputs. "
                    f"tensor_error={type(e_tensor).__name__}: {e_tensor}; "
                    f"callable_error={type(e_callable).__name__}: {e_callable}"
                ) from e_callable
    return evals, evecs, backend


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--input_npz", required=True, help="SciPy sparse .npz matrix (2D SPPMI)")
    ap.add_argument("--out_dir", required=True, help="Output directory")

    ap.add_argument("--threshold", type=float, default=None, help="Drop entries < threshold")
    ap.add_argument("--drop_diagonal", action="store_true", help="Set diagonal of A to 0")
    ap.add_argument("--symmetrize", action="store_true", help="A := (A + A.T)/2")

    ap.add_argument("--dtype", choices=["float32", "float64"], default="float64",
                    help="Compute dtype. float64 is more stable for eigsh; GPU often prefers float32.")
    ap.add_argument("--save_A", action="store_true", help="Also save processed A as A_csr.npz")

    # eigensolver selection
    ap.add_argument("--solver", choices=["eigsh", "torch_lobpcg"], default="eigsh",
                    help="eigsh=CPU ARPACK; torch_lobpcg=GPU/CPU via PyTorch LOBPCG.")
    ap.add_argument("--device", default="cpu",
                    help="cpu, cuda, cuda:0, cuda:1 ... (used when solver=torch_lobpcg)")

    # eigsh params (CPU)
    ap.add_argument("--k", type=int, default=128, help="Number of eigenvectors/eigenvalues")
    ap.add_argument("--which", default="LA", help="ARPACK selector (eigsh only): LA, LM, SA, ...")
    ap.add_argument("--tol", type=float, default=1e-3, help="Convergence tolerance (eigsh) / lobpcg tol")
    ap.add_argument("--maxiter", type=int, default=2000, help="Max iterations (eigsh)")
    ap.add_argument("--ncv", type=int, default=None, help="Krylov subspace size (eigsh)")

    # lobpcg params (torch)
    ap.add_argument("--lobpcg_niter", type=int, default=200,
                    help="Max iterations for torch.lobpcg (torch_lobpcg solver)")
    ap.add_argument("--lobpcg_init", choices=["randn", "rand"], default="randn",
                    help="Initialization for LOBPCG")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for LOBPCG init")

    # embeddings
    ap.add_argument("--embed", choices=["evecs", "sqrt_evals"], default="sqrt_evals",
                    help="How to turn eigenpairs into embeddings")
    ap.add_argument("--vocab_csv", default="",
                    help="Optional vocab CSV (from E) to write an embeddings CSV.")
    ap.add_argument("--output_csv", default="",
                    help="Optional embeddings CSV output path.")
    ap.add_argument("--emb_prefix", default="emb_",
                    help="Prefix for embedding columns in CSV output.")
    ap.add_argument("--cui_col", default="cui",
                    help="CUI column name in vocab CSV.")
    ap.add_argument("--out_cui_col", default="umls_top_cui",
                    help="CUI column name in embeddings CSV.")

    # threading (CPU-side prep)
    ap.add_argument("--threads", type=int, default=None,
                    help="Set CPU threads for numpy/scipy/torch (best-effort)")

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    # ---------------------------
    # Resource tracking start
    # ---------------------------
    wall_start = time.perf_counter()
    cpu0 = _cpu_usage_snapshot()
    gpu_id = _infer_gpu_id_from_device(args.device)
    gpu0 = _nvidia_smi_snapshot(gpu_id=gpu_id)

    if args.threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)

    dtype_np = np.float32 if args.dtype == "float32" else np.float64

    # Load 2D sparse matrix
    A = sp.load_npz(args.input_npz)
    if not sp.isspmatrix(A):
        raise TypeError("Loaded object is not a SciPy sparse matrix.")
    A = A.tocsr().astype(dtype_np, copy=False)

    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Expected a square 2D matrix. Got {A.shape}.")

    print(f"[INFO] Loaded A: shape={A.shape} nnz={A.nnz:,} dtype={A.dtype}")

    # Optional threshold
    if args.threshold is not None:
        A = threshold_scipy_sparse(A, args.threshold)
        print(f"[INFO] After threshold: nnz={A.nnz:,}")

    # Optional drop diagonal
    if args.drop_diagonal:
        A.setdiag(0)
        A.eliminate_zeros()
        print(f"[INFO] After drop_diagonal: nnz={A.nnz:,}")

    # Optional symmetrize
    if args.symmetrize:
        A = (A + A.T) * 0.5
        A.eliminate_zeros()
        A = A.tocsr()
        print(f"[INFO] After symmetrize: nnz={A.nnz:,}")

    if args.save_A:
        outA = os.path.join(args.out_dir, "A_csr.npz")
        sp.save_npz(outA, A)
        print(f"[OK] Saved processed A -> {outA}")

    meta: Dict[str, Any] = {
        "input_npz": args.input_npz,
        "A_shape": [int(A.shape[0]), int(A.shape[1])],
        "A_nnz": int(A.nnz),
        "threshold": args.threshold,
        "drop_diagonal": bool(args.drop_diagonal),
        "symmetrize": bool(args.symmetrize),
        "dtype": args.dtype,
        "solver": args.solver,
        "device": args.device,
        "embedding": args.embed,
    }

    # ---------------------------
    # Solve eigenproblem
    # ---------------------------
    if args.solver == "eigsh":
        if args.device != "cpu":
            raise SystemExit("solver=eigsh is CPU-only. Use --solver torch_lobpcg for GPU.")
        meta["eigsh"] = {
            "k": args.k,
            "which": args.which,
            "tol": args.tol,
            "maxiter": args.maxiter,
            "ncv": args.ncv,
        }

        print("[INFO] Running eigsh (CPU, ARPACK)...")
        evals, evecs = eigsh(
            A,
            k=args.k,
            which=args.which,
            tol=args.tol,
            maxiter=args.maxiter,
            ncv=args.ncv,
            return_eigenvectors=True,
        )

        # Sort descending
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]

    elif args.solver == "torch_lobpcg":
        # GPU/CPU via torch
        try:
            import torch
        except Exception as e:
            raise SystemExit(
                "PyTorch is required for solver=torch_lobpcg. "
                "Install torch (with CUDA build if using GPU). "
                f"Import error: {e}"
            )

        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA device requested but torch.cuda.is_available() is False.")

        if args.threads is not None:
            try:
                torch.set_num_threads(args.threads)
            except Exception:
                pass

        torch_dtype = torch.float32 if args.dtype == "float32" else torch.float64
        meta["torch_lobpcg"] = {
            "k": args.k,
            "largest": True,
            "tol": args.tol,
            "niter": args.lobpcg_niter,
            "init": args.lobpcg_init,
            "seed": args.seed,
            "torch_dtype": str(torch_dtype),
        }

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device=device)

        print(f"[INFO] Converting SciPy sparse -> torch sparse COO on {device}...")
        A_torch = scipy_to_torch_sparse_coo(A, device=device, dtype=torch_dtype)
        print(f"[INFO] torch sparse A: shape={tuple(A_torch.shape)} nnz={A_torch._nnz():,} dtype={A_torch.dtype}")

        print("[INFO] Running torch.lobpcg...")
        # Require the sparse tensor to be placed on requested device.
        if A_torch.device.type != device.type:
            raise SystemExit(
                f"Device mismatch: requested {device}, but A_torch is on {A_torch.device}."
            )
        if device.type == "cuda":
            req_idx = device.index if device.index is not None else 0
            got_idx = A_torch.device.index if A_torch.device.index is not None else 0
            if got_idx != req_idx:
                raise SystemExit(
                    f"CUDA device mismatch: requested cuda:{req_idx}, got cuda:{got_idx}."
                )

        evals_t, evecs_t, lobpcg_backend = torch_lobpcg_topk(
            A_torch,
            k=args.k,
            largest=True,
            tol=args.tol,
            niter=args.lobpcg_niter,
            seed=args.seed,
            init=args.lobpcg_init,
        )
        meta["torch_lobpcg"]["backend"] = lobpcg_backend
        print(
            f"[INFO] torch.lobpcg backend={lobpcg_backend} "
            f"evals_device={evals_t.device} evecs_device={evecs_t.device}"
        )
        if device.type == "cuda" and (evals_t.device.type != "cuda" or evecs_t.device.type != "cuda"):
            raise SystemExit(
                "CUDA was requested, but torch.lobpcg outputs are not on CUDA."
            )

        # Sort descending
        order = torch.argsort(evals_t, descending=True)
        evals_t = evals_t[order]
        evecs_t = evecs_t[:, order]

        # Move back to CPU numpy for saving
        evals = evals_t.detach().cpu().numpy()
        evecs = evecs_t.detach().cpu().numpy()

    else:
        raise SystemExit(f"Unknown solver: {args.solver}")

    # ---------------------------
    # Embeddings + save
    # ---------------------------
    emb = make_embeddings(evals, evecs, method=args.embed)

    np.save(os.path.join(args.out_dir, "eigvals.npy"), evals)
    np.save(os.path.join(args.out_dir, "eigvecs.npy"), evecs)
    np.save(os.path.join(args.out_dir, "embeddings.npy"), emb)

    save_json(meta, os.path.join(args.out_dir, "meta.json"))

    print(f"[OK] Saved eigvals/eigvecs/embeddings -> {args.out_dir}")
    print(f"[INFO] Top-5 eigenvalues: {evals[:5]}")
    print(f"[INFO] Embeddings shape: {emb.shape}")

    # Optional: write embeddings CSV for downstream steps (e.g., G)
    if args.output_csv:
        if not args.vocab_csv:
            raise SystemExit("output_csv requires --vocab_csv")
        import pandas as pd

        vocab_df = pd.read_csv(args.vocab_csv)
        if args.cui_col not in vocab_df.columns:
            raise SystemExit(f"vocab CSV missing column '{args.cui_col}': {args.vocab_csv}")
        if len(vocab_df) != emb.shape[0]:
            raise SystemExit(
                f"vocab rows ({len(vocab_df)}) != embeddings rows ({emb.shape[0]})"
            )

        emb_cols = [f"{args.emb_prefix}{i}" for i in range(emb.shape[1])]
        out_df = pd.DataFrame(emb, columns=emb_cols)
        out_df.insert(0, args.out_cui_col, vocab_df[args.cui_col].astype(str).values)
        out_df.to_csv(args.output_csv, index=False)
        print(f"[OK] Saved embeddings CSV: {args.output_csv}")

    # ---------------------------
    # Resource tracking end
    # ---------------------------
    wall_end = time.perf_counter()
    cpu1 = _cpu_usage_snapshot()
    gpu1 = _nvidia_smi_snapshot(gpu_id=gpu_id)

    wall_s = wall_end - wall_start
    cpu_user = cpu1["user_s"] - cpu0["user_s"]
    cpu_sys = cpu1["sys_s"] - cpu0["sys_s"]
    cpu_total = cpu_user + cpu_sys

    cores = int(os.environ.get("SLURM_CPUS_PER_TASK", "0") or 0)
    if cores <= 0:
        cores = os.cpu_count() or 1

    est_cpu_util = _estimate_cpu_util_pct(cpu_total, wall_s, cores)

    print("\n================ Resource report ================")
    print(f"[RUNTIME] wall_s={wall_s:,.3f}")
    print(f"[CPU] user_s={cpu_user:,.3f} sys_s={cpu_sys:,.3f} total_s={cpu_total:,.3f}")
    print(f"[CPU] maxrss_mb={cpu1['maxrss_mb']:,.2f} (process peak RSS)")
    print(f"[CPU] cores_ref={cores} est_cpu_util_pct={est_cpu_util:,.2f}")

    if args.solver == "torch_lobpcg":
        try:
            import torch
            if torch.cuda.is_available() and _infer_gpu_id_from_device(args.device) is not None:
                dev = torch.device(args.device)
                peak_alloc = torch.cuda.max_memory_allocated(device=dev)
                peak_resv = torch.cuda.max_memory_reserved(device=dev)
                print(f"[GPU] torch peak_alloc={_fmt_mb(peak_alloc)} peak_reserved={_fmt_mb(peak_resv)}")
        except Exception:
            pass

    if gpu0 is not None or gpu1 is not None:
        print("[GPU] nvidia-smi snapshots (start/end):")
        print(json.dumps({"start": gpu0, "end": gpu1}, indent=2))
    else:
        print("[GPU] nvidia-smi not available (or no GPU visible).")

    print("=================================================\n")


if __name__ == "__main__":
    main()
