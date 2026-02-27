import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


@dataclass
class Candidate:
    cui: str
    term: str
    semantic_type: str
    score: float


class UMLSIndex:
    """FAISS-based UMLS term linker built on SapBERT-style encoders.

    Parameters
    ----------
    model_name: str
        HF model id or local path (default: 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext').
    dictionary_path: str
        Path to a text file with lines formatted as `CUI||term||semantic_type`.
    cache_dir: str
        Directory to store/reuse dictionary embeddings (default: './cache').
    device: str
        'cuda' or 'cpu'. If 'cuda' but GPU is unavailable, falls back to CPU.
    fp16: bool
        Use half precision for encoder forward pass when device is cuda.
    use_faiss_gpu: bool
        Build/search FAISS index on GPU when available.
    use_all_gpus: bool
        When True and multiple GPUs exist, spread FAISS index across all GPUs.
    use_ivf: bool
        Use IVF flat index when dictionary is large (helps memory/speed).
    ivf_threshold: int
        Minimum vocabulary size to trigger IVF indexing.
    local_files_only: bool
        If True, transformers loads model/tokenizer without touching the network.
    """

    def __init__(
        self,
        model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        dictionary_path: str = "./umls_dictionary.txt",
        cache_dir: str = "./cache",
        device: str = "cuda",
        fp16: bool = False,
        use_faiss_gpu: bool = True,
        use_all_gpus: bool = False,
        gpu_id: int = 0,
        use_ivf: bool = True,
        ivf_threshold: int = 50_000,
        local_files_only: bool = False,
    ):
        self.model_name = model_name
        self.dictionary_path = Path(dictionary_path)
        self.cache_dir = Path(cache_dir)
        self.device = device if device in {"cuda", "cpu"} else "cpu"
        self.fp16 = fp16
        self.use_faiss_gpu = use_faiss_gpu
        self.use_all_gpus = use_all_gpus
        self.gpu_id = gpu_id
        self.use_ivf = use_ivf
        self.ivf_threshold = ivf_threshold
        self.local_files_only = local_files_only

        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "[%(asctime)s][%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # Lazy initialized attributes
        self.tokenizer: Optional[AutoTokenizer] = None
        self.encoder: Optional[AutoModel] = None
        self.term_list: List[str] = []
        self.term_semantic: Dict[str, Tuple[str, str]] = {}
        self.term_embeds: Optional[torch.Tensor] = None
        self.index: Optional[faiss.Index] = None

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def load(self, batch_size: int = 256) -> None:
        """Load dictionary, encode terms (with cache), and build FAISS index."""
        self._load_dictionary()
        self._load_model()
        self._prepare_embeddings(batch_size=batch_size)
        self._prepare_index()

    def search(self, terms: List[str], top_k: int = 1, batch_size: int = 256) -> List[List[Candidate]]:
        """Return top-k candidates for each term.

        Returns a list with the same length as `terms`; each entry is a list of
        :class:`Candidate` objects ordered by similarity.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call load() first.")

        cleaned_terms, keep_indices = self._clean_terms(terms)
        if not cleaned_terms:
            return [[] for _ in terms]

        query_embeds = self._embed(cleaned_terms, batch_size=batch_size)
        query_np = query_embeds.cpu().numpy().astype(np.float32)

        D, I = self.index.search(query_np, top_k)

        results: List[List[Candidate]] = []
        for d_row, i_row in zip(D, I):
            cands: List[Candidate] = []
            for score, idx in zip(d_row, i_row):
                if idx < 0 or idx >= len(self.term_list):
                    continue
                term = self.term_list[idx]
                cui, sty = self.term_semantic[term]
                cands.append(Candidate(cui=cui, term=term, semantic_type=sty, score=float(score)))
            results.append(cands)

        # Re-expand to original input length, preserving None rows
        expanded: List[List[Candidate]] = []
        result_iter = iter(results)
        for i in range(len(terms)):
            if i in keep_indices:
                expanded.append(next(result_iter))
            else:
                expanded.append([])
        return expanded

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------
    def _clean_terms(self, terms: List[str]) -> Tuple[List[str], List[int]]:
        cleaned = []
        idx = []
        for i, t in enumerate(terms):
            if t is None:
                continue
            t_str = str(t).strip()
            if not t_str:
                continue
            cleaned.append(t_str)
            idx.append(i)
        return cleaned, idx

    def _load_dictionary(self) -> None:
        if not self.dictionary_path.exists():
            raise FileNotFoundError(f"Dictionary not found: {self.dictionary_path}")

        terms: List[str] = []
        semantic: Dict[str, Tuple[str, str]] = {}
        with self.dictionary_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("||")
                if len(parts) != 3:
                    continue
                cui, term, sty = parts
                terms.append(term)
                semantic[term] = (cui, sty)

        # Sort once for deterministic indexing
        unique_terms = sorted(set(terms))
        self.term_list = unique_terms
        self.term_semantic = semantic
        self.logger.info(f"Loaded {len(self.term_list):,} terms from {self.dictionary_path}")

    def _load_model(self) -> None:
        self.logger.info(f"Loading encoder: {self.model_name}")
        if self.local_files_only:
            self.logger.info("Transformers offline mode enabled (local files only).")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
            do_lower_case=True,
            local_files_only=self.local_files_only,
        )
        self.encoder = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )

        use_cuda = self.device == "cuda" and torch.cuda.is_available()
        if not use_cuda:
            self.device = "cpu"
        self.encoder.to(self.device)
        if use_cuda and self.fp16:
            self.encoder.half()
        self.encoder.eval()
        self.logger.info(f"Encoder loaded on {self.device} (fp16={self.fp16 and use_cuda})")

    def _cache_key(self) -> Path:
        sig = f"{self.dictionary_path.resolve()}::{self.dictionary_path.stat().st_mtime}::{self.model_name}"
        hashed = hashlib.md5(sig.encode()).hexdigest()[:10]
        fname = f"umls_{hashed}.pt"
        return self.cache_dir / fname

    def _prepare_embeddings(self, batch_size: int = 256) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self._cache_key()

        if cache_file.exists():
            self.logger.info(f"Loading cached dictionary embeddings from {cache_file}")
            payload = torch.load(cache_file, map_location="cpu")
            self.term_embeds = payload["embeds"]
            self.term_list = payload["terms"]
            self.logger.info(f"Loaded {self.term_embeds.shape[0]:,} cached embeddings")
            # Move to device if needed for downstream operations
            return

        self.logger.info(f"Embedding {len(self.term_list):,} terms (batch={batch_size})")
        embeds = []
        with torch.no_grad():
            for start in tqdm(range(0, len(self.term_list), batch_size)):
                batch_terms = self.term_list[start:start + batch_size]
                # Lower-case for stable encoding, mirroring search-time prep
                batch_terms_lower = [t.lower() for t in batch_terms]
                tokenized = self.tokenizer.batch_encode_plus(
                    batch_terms_lower,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=25,
                    padding="max_length",
                    return_tensors="pt",
                )
                tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
                if self.fp16 and self.device == "cuda":
                    tokenized = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in tokenized.items()}

                output = self.encoder(**tokenized).last_hidden_state[:, 0, :]
                output = output / torch.norm(output, p=2, dim=-1, keepdim=True)
                embeds.append(output.cpu())
                if self.device == "cuda":
                    torch.cuda.empty_cache()

        self.term_embeds = torch.cat(embeds, dim=0)
        torch.save({"embeds": self.term_embeds, "terms": self.term_list}, cache_file)
        self.logger.info(f"Saved embeddings to cache: {cache_file}")

    def _prepare_index(self) -> None:
        if self.term_embeds is None:
            raise RuntimeError("Embeddings not prepared.")

        embeddings_np = self.term_embeds.cpu().numpy().astype(np.float32)
        dim = embeddings_np.shape[1]

        use_ivf_current = self.use_ivf and len(self.term_list) > self.ivf_threshold
        if use_ivf_current:
            nlist = min(4096, max(int(len(self.term_list) / 50), 256))
            quantizer = faiss.IndexFlatIP(dim)
            index_cpu = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index_cpu.train(embeddings_np)
        else:
            index_cpu = faiss.IndexFlatIP(dim)

        index_cpu.add(embeddings_np)

        if self.use_faiss_gpu and torch.cuda.is_available():
            self.logger.info("Moving FAISS index to GPU(s)")
            if self.use_all_gpus and faiss.get_num_gpus() > 1:
                self.index = faiss.index_cpu_to_all_gpus(index_cpu)
            else:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, self.gpu_id, index_cpu)
        else:
            self.index = index_cpu
            if self.use_faiss_gpu and not torch.cuda.is_available():
                self.logger.warning("GPU requested for FAISS but CUDA not available; using CPU index.")

        # For IVF index, tune nprobe for better recall
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = min(32, getattr(self.index, "nlist", 32))

        self.logger.info(
            "FAISS index ready: %s | size=%s | dim=%s | IVF=%s",
            self.index.__class__.__name__,
            len(self.term_list),
            dim,
            use_ivf_current,
        )

    def _embed(self, terms: List[str], batch_size: int = 256) -> torch.Tensor:
        """Embed query terms using the loaded encoder."""
        if self.encoder is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load().")

        embeds = []
        with torch.no_grad():
            for start in range(0, len(terms), batch_size):
                batch_terms = [t.lower() for t in terms[start:start + batch_size]]
                tokenized = self.tokenizer.batch_encode_plus(
                    batch_terms,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=25,
                    padding="max_length",
                    return_tensors="pt",
                )
                tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
                if self.fp16 and self.device == "cuda":
                    tokenized = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in tokenized.items()}

                output = self.encoder(**tokenized).last_hidden_state[:, 0, :]
                output = output / torch.norm(output, p=2, dim=-1, keepdim=True)
                embeds.append(output.cpu())
                if self.device == "cuda":
                    torch.cuda.empty_cache()

        if embeds:
            return torch.cat(embeds, dim=0)
        return torch.zeros((0, self.encoder.config.hidden_size))


__all__ = ["UMLSIndex", "Candidate"]
