import json
import os
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from .retriever import Candidate, UMLSIndex


def _clip_confidence(score: Optional[float]) -> Optional[float]:
    if score is None:
        return None
    return max(0.0, min(1.0, float(score)))


def _candidate_to_dict(cand: Candidate) -> Dict[str, object]:
    return {
        "cui": cand.cui,
        "term": cand.term,
        "semantic_type": cand.semantic_type,
        "confidence": _clip_confidence(cand.score),
    }


def _load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, encoding="latin-1")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="cp1252")


def _normalize_series(
    values: List[str],
    retriever: UMLSIndex,
    top_k: int,
    threshold: float,
    batch_size: int,
):
    # Deduplicate normalized strings to cut embedding cost on repeats
    norm_to_original_indices: Dict[str, List[int]] = {}
    norm_terms: List[str] = []
    for i, term in enumerate(values):
        if term is None or str(term).strip() == "":
            continue
        norm = str(term).strip().lower()
        if norm not in norm_to_original_indices:
            norm_terms.append(norm)
            norm_to_original_indices[norm] = []
        norm_to_original_indices[norm].append(i)

    norm_results = retriever.search(norm_terms, top_k=top_k, batch_size=batch_size) if norm_terms else []
    norm_map = {term: res for term, res in zip(norm_terms, norm_results)}

    normalized = []
    for term in values:
        if term is None or str(term).strip() == "":
            normalized.append(
                {
                    "top_term": None,
                    "top_cui": None,
                    "top_confidence": None,
                    "top_semantic_type": None,
                    "kept": [],
                    "warn_below_threshold": False,
                }
            )
            continue

        cands = norm_map.get(str(term).strip().lower(), [])
        filtered = [c for c in cands if c.score >= threshold]
        kept = filtered if filtered else cands[:1]  # keep best even if below threshold

        warn_flag = bool(kept and kept[0].score < threshold)
        top = kept[0] if kept else None
        normalized.append(
            {
                "top_term": top.term if top else None,
                "top_cui": top.cui if top else None,
                "top_confidence": _clip_confidence(top.score if top else None),
                "top_semantic_type": top.semantic_type if top else None,
                "kept": [_candidate_to_dict(c) for c in kept],
                "warn_below_threshold": warn_flag,
            }
        )

    return normalized


def normalize_file(
    input_path: str,
    output_path: str,
    retriever: UMLSIndex,
    entity_column: str,
    top_k: int = 1,
    threshold: float = 0.35,
    batch_size: int = 256,
) -> Tuple[str, int]:
    """Normalize a single CSV file and write enriched CSV.

    Returns (output_path, row_count).
    """
    input_p = Path(input_path)
    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)

    df = _load_csv(input_p)
    if entity_column not in df.columns:
        raise KeyError(f"Column '{entity_column}' not found in {input_path}")

    values = df[entity_column].tolist()
    normalized = _normalize_series(values, retriever, top_k, threshold, batch_size)

    df["umls_top_term"] = [item["top_term"] for item in normalized]
    df["umls_top_cui"] = [item["top_cui"] for item in normalized]
    df["umls_top_confidence"] = [item["top_confidence"] for item in normalized]
    df["umls_top_semantic_type"] = [item["top_semantic_type"] for item in normalized]
    df["umls_candidates"] = [
        json.dumps(item["kept"], ensure_ascii=False) if item["kept"] else "[]"
        for item in normalized
    ]
    df["umls_below_threshold_warning"] = [item["warn_below_threshold"] for item in normalized]

    df.to_csv(output_p, index=False)
    return str(output_p), len(df)


def normalize_folder(
    input_dir: str,
    output_dir: str,
    retriever: UMLSIndex,
    entity_column: str,
    top_k: int = 1,
    threshold: float = 0.35,
    batch_size: int = 256,
    pattern: str = "*.csv",
    skip_existing: bool = False,
) -> List[Tuple[str, int]]:
    """Normalize every CSV under input_dir matching pattern.

    Returns list of (output_path, row_count) tuples.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    files = sorted(glob(str(input_path / pattern)))
    results = []

    for file in tqdm(files, desc="files"):
        rel = Path(file).relative_to(input_path)
        out_file = output_path / rel
        if skip_existing and out_file.exists():
            continue
        out_file = out_file.with_suffix(".normalized.csv")
        results.append(
            normalize_file(
                input_path=file,
                output_path=str(out_file),
                retriever=retriever,
                entity_column=entity_column,
                top_k=top_k,
                threshold=threshold,
                batch_size=batch_size,
            )
        )
    return results


__all__ = ["normalize_file", "normalize_folder"]
