# UMLS Term Normalizer

Batch normalize raw entity strings to UMLS concepts inside large CSV collections. The workflow mirrors the SapBERT + FAISS recipe from `language-into-clinical-data`: given a folder of CSV files with one column containing free-text mentions, the tool appends the best matching UMLS term, CUI, semantic type, candidate list, similarity score, and warning flags to every row.

## Highlights
- SapBERT encoder + FAISS vector search with CPU, single-GPU, or multi-GPU execution.
- Top-k normalization with per-candidate similarity scores.
- Threshold filtering that keeps the best candidate even when all fall below the cutoff (and marks a warning).
- Production-friendly for large vocabularies:
  - Dictionary embeddings are cached on disk and reused automatically.
  - Optional IVF index reduces memory and latency for very large dictionaries.
  - Repeated mentions are deduplicated before querying.
- Traceable results: persist the full candidate list (JSON) and semantic types for every mention.

## Repository Layout
```
umls-normalizer/
├── requirements.txt          # Python deps (GPU builds should swap faiss-cpu → faiss-gpu)
├── README.md                 # You are here
└── umls_normalizer/
    ├── cli.py                # Command-line entrypoint
    ├── retriever.py          # SapBERT encoder + FAISS indexing logic
    └── normalizer.py         # CSV I/O + normalization pipeline
```

## Environment Setup (uv recommended)
**CPU / lightweight (Python 3.12 + faiss-cpu)**
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

**Full GPU (encoder + FAISS on CUDA)**
```bash
uv python install 3.10
uv venv --python 3.10 .venv310
source .venv310/bin/activate
uv pip install -r requirements.txt              # installs torch/cu12
uv pip uninstall --python .venv310/bin/python faiss-cpu
UV_LINK_MODE=copy uv pip install --python .venv310/bin/python faiss-gpu==1.7.2
# if numpy 2.x conflicts occur:
uv pip install --python .venv310/bin/python 'numpy>=1.24,<2.0'
```
> The official `faiss-gpu` wheel currently stops at Python 3.10, so GPU FAISS requires a 3.10 virtualenv.

## Data Prerequisites
Prepare a UMLS dictionary text file where each line is `CUI||term||semantic_type`. The repo ships with a toy example at `examples/umls_dictionary.txt`; replace it with your production dictionary for real workloads. On the O2 cluster, the shared snapshot lives at `/n/data1/hsph/biostat/celehs/lab/SHARE/From_Zongxin/language-into-clinical-data/umls_dictionary.txt` so every job can reuse the same vocabulary without copying per-user.

## Quick Start
### Single file (GPU encoding + CPU FAISS)
```bash
python -m umls_normalizer.cli \
  --mode file \
  --input-file /path/to/input.csv \
  --output-dir /path/to/out \
  --entity-column mention \
  --dictionary /path/to/umls_dictionary.txt \
  --model-name cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
  --top-k 3 \
  --threshold 0.35 \
  --device cuda --fp16 \
  --no-faiss-gpu
```

### Folder batch (multi-GPU)
```bash
python -m umls_normalizer.cli \
  --mode folder \
  --input-dir /data/raw_csvs \
  --output-dir /data/normalized_csvs \
  --entity-column entity \
  --dictionary /path/to/umls_dictionary.txt \
  --top-k 5 --threshold 0.4 \
  --device cuda --use-all-gpus \
  --use-ivf --ivf-threshold 80000 \
  --pattern "*.csv" --skip-existing
```

### CPU-only
```bash
python -m umls_normalizer.cli \
  --mode folder \
  --input-dir /data/raw_csvs \
  --output-dir /data/normalized_csvs \
  --entity-column entity \
  --dictionary /path/to/umls_dictionary.txt \
  --device cpu --no-faiss-gpu \
  --dict-batch-size 128 --query-batch-size 128
```

### Slurm / `srun` (gpu_dia partition, GPU encoding + CPU FAISS)
```bash
srun -p gpu_dia -N1 -n1 --gres=gpu:1 --time=00:05:00 bash -lc '
  cd /path/to/umls-normalizer &&
  source .venv/bin/activate &&
  python -m umls_normalizer.cli \
    --mode folder \
    --input-dir examples/input \
    --output-dir examples/output \
    --entity-column entity \
    --dictionary examples/umls_dictionary.txt \
    --device cuda --no-faiss-gpu \
    --top-k 3 --threshold 0.2 \
    --pattern "*.csv"'
```

### Slurm / `srun` (gpu_dia, full GPU pipeline)
```bash
srun -p gpu_dia -N1 -n1 --gres=gpu:1 --mem=16G --time=00:10:00 bash -lc '
  cd /path/to/umls-normalizer &&
  source .venv310/bin/activate &&
  python -m umls_normalizer.cli \
    --mode folder \
    --input-dir /path/to/input_dir \
    --output-dir /path/to/output_dir \
    --entity-column entity \
    --dictionary /path/to/umls_dictionary.txt \
    --device cuda \
    --top-k 5 --threshold 0.35 \
    --pattern "*.csv"'
```

## Offline SapBERT Loading
Some secure clusters have no outbound network access. The CLI now supports transformer offline mode so SapBERT can be loaded entirely from local files.

1. **Download SapBERT ahead of time** (any machine with internet):
   ```bash
   mkdir -p models && cd models
   huggingface-cli download cambridgeltl/SapBERT-from-PubMedBERT-fulltext --local-dir sapbert_full
   # or: git lfs clone https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext sapbert_full
   tar -czf sapbert_full.tar.gz sapbert_full
   ```
2. **Transfer the archive** (`sapbert_full.tar.gz`) to the offline server and extract it, e.g. `/opt/models/sapbert_full`.
3. **Point the CLI to the local snapshot** and enable offline mode:
   ```bash
   export TRANSFORMERS_OFFLINE=1            # optional but recommended safeguard
   python -m umls_normalizer.cli \
     --mode folder \
     --input-dir /data/raw \
     --output-dir /data/normalized \
     --entity-column entity \
     --dictionary /data/umls_dictionary.txt \
     --model-name /opt/models/sapbert_full \
     --offline \
     --device cuda --no-faiss-gpu
   ```
   - `--model-name` accepts either a Hugging Face model id or any local directory containing `config.json`, `pytorch_model.bin`, tokenizer files, etc.
   - `--offline` sets `local_files_only=True` internally so transformers never call the Hugging Face Hub.
   - Leave `TRANSFORMERS_OFFLINE=1` in the environment if you want transformers to error out whenever a file is missing instead of silently attempting network access.

## Output Columns
Each normalized CSV receives the following columns:
- `umls_top_term`: best matching canonical term.
- `umls_top_cui`: CUI of the best match.
- `umls_top_confidence`: cosine similarity score (vector dot product on normalized embeddings).
- `umls_top_semantic_type`: semantic type label.
- `umls_candidates`: JSON list of candidate objects (term, CUI, semantic type, confidence) filtered by the threshold.
- `umls_below_threshold_warning`: `True` if the retained top-1 candidate falls below `--threshold`.

## Performance Tips
- **Caching**: dictionary embeddings are persisted under `--cache-dir`; reuse the same directory to skip recomputing on subsequent runs.
- **Multi-GPU FAISS**: add `--use-all-gpus` to shard the index across all visible GPUs. Alternatively, spawn multiple processes each handling a subset of files.
- **IVF index**: enable `--use-ivf` when vocabularies are very large and tune `--ivf-threshold` / FAISS parameters for the best trade-off.
- **Batch sizes**: adjust `--dict-batch-size` and `--query-batch-size` based on available VRAM/RAM.

## Troubleshooting
- **Missing column**: ensure `--entity-column` exactly matches the CSV header (case-sensitive).
- **Empty cells**: blank strings or nulls are skipped and emit empty candidate lists without warnings.
- **Low scores**: increase `--threshold` to filter more aggressively; the top candidate is still retained (with `umls_below_threshold_warning=True`) so records are never empty.
- **FAISS GPU unavailable**: the CLI warns and falls back to CPU if CUDA is missing; double-check CUDA visibility or install `faiss-gpu`.
