# A–J Tuning Parameter Codebook

This codebook documents **tuning parameters** (behavioral knobs) in the current A–J scripts. For completeness, each section also lists **schema/path parameters** that change inputs/outputs or column mappings but do not tune model behavior.

Conventions:
- **Tuning** = affects filtering, thresholds, model behavior, or optimization.
- **Schema/Path** = input/output paths or column name mappings.

---

## Pipeline Orchestrator: `genie_pipeline.sh`

`genie_pipeline.sh` is configured through environment variables (not CLI flags). The tables below describe all user-overridable parameters currently read by the script.

### How to pass values

- `VAR=value sbatch genie_pipeline.sh`
- `sbatch --export=ALL,VAR1=value,VAR2=value genie_pipeline.sh`
- `VAR=value bash genie_pipeline.sh` (non-Slurm testing)

### Important path behavior

- Steps A-D are invoked with `--base-dir "$GENIE_BASE"` plus mostly `*_REL` paths. Their paired `*_ABS` variables are used mainly for validation/logging.
- If you override an `*_ABS` variable, keep it consistent with the corresponding `*_REL` variable to avoid precheck/runtime mismatches.

### Directory map (required vs auto-created)

| Directory (resolved) | How it is set | Auto-created by pipeline? | Notes |
|---|---|---|---|
| `${ENV_BASE}` | `ENV_BASE` | No (assume existing/writable parent) | Parent location for conda envs. |
| `${ENV_CPU_DIR}` | `ENV_CPU_DIR` | Yes (via `conda create -p`) | CPU conda env path. |
| `${ENV_GPU_DIR}` | `ENV_GPU_DIR` | Yes (via `conda create -p`) | GPU conda env path. |
| `${UMLS_DIR}` | `UMLS_DIR` | No | Must exist and be accessible; script `cd`s here. |
| `${UMLS_NORM_ROOT}` | `UMLS_NORM_ROOT` | No | Must contain importable `umls_normalizer` package. |
| `${INPUT_UMLS_DER}` | `${UMLS_DIR}/${INPUT_UMLS_REL}` | Yes (`mkdir -p`) | Step 3 input folder for UMLS run. |
| `${OUTPUT_UMLS_DER}` | `${UMLS_DIR}/${OUTPUT_UMLS_REL}` | Yes (`mkdir -p`) | Step 3 output folder for UMLS run. |
| `${GENIE_BASE}` | `GENIE_BASE` | Indirectly (`mkdir -p` on subdirs) | Project root for Data/Output/Log and default script/input paths. |
| `${GENIE_BASE}/Data` | Derived from `GENIE_BASE` | Yes (`mkdir -p`) | Main data folder (inputs + many outputs). |
| `${GENIE_BASE}/Output` | Derived from `GENIE_BASE` | Yes (`mkdir -p`) | Parent of per-run output folders. |
| `${GENIE_BASE}/Log` | Derived from `GENIE_BASE` | Yes (`mkdir -p`) | Time/stdout/stderr and GPU monitor logs. |
| `${SCRIPTS_DIR}` | `SCRIPTS_DIR` | No | Must contain A-J scripts unless each `SCRIPT_DER_*` is overridden to valid paths. |
| `${PIPE_OUT_DIR}` | `PIPE_OUT_DIR` | Yes (`mkdir -p`) | Per-run output root (typically `${GENIE_BASE}/Output/run_${RUN_TAG}`). |
| `${PRE6_OUT_DIR}` | `PRE6_OUT_DIR` | Yes (`mkdir -p`) | Parent of Step 5 vocab output (`E_VOCAB_ABS`). |
| `$(dirname "${SPPMI_NPZ_ABS}")` | From `SPPMI_NPZ_ABS` | Yes (`ensure_dir`) | Step 5 matrix output directory. |
| `$(dirname "${E_VOCAB_ABS}")` | From `E_VOCAB_ABS` | Yes (`ensure_dir`) | Step 5 vocab output directory. |
| `${F_OUT_DIR}` | `F_OUT_DIR` | Yes (`ensure_dir`) | Step 6 decomposition output directory. |
| `$(dirname "${EMBED_CSV_ABS}")` | From `EMBED_CSV_ABS` | Yes (`ensure_dir`) | Step 6 embeddings CSV parent directory. |
| `$(dirname "${G_PAIRS_ABS}")` | From `G_PAIRS_ABS` | Yes (`ensure_dir`) | Step 7 pairs output directory. |
| `$(dirname "${H_OUT_ABS}")` | From `H_OUT_ABS` | Yes (`ensure_dir`) | Step 8 enriched-pairs output directory. |
| `$(dirname "${I_OUT_ABS}")` | From `I_OUT_ABS` | Yes (`ensure_dir`) | Step 9 scoring output directory. |
| `$(dirname "${J_OUT_ABS}")` | From `J_OUT_ABS` | Yes (`ensure_dir`) | Step 10 final output directory. |

### Minimal directory bootstrap (from scratch)

Use this when setting up a new workspace on cluster/storage. Adjust paths first.

```bash
# 1) Choose base locations
export GENIE_BASE=/path/to/GENIE_Analysis_Consolidated_0121
export UMLS_DIR=/path/to/umls_workspace
export UMLS_NORM_ROOT="$UMLS_DIR/umls-normalizer-main"
export SCRIPTS_DIR="$GENIE_BASE/Scripts_Refined_0121"

# 2) Create required directory skeleton
mkdir -p "$GENIE_BASE/Data" "$GENIE_BASE/Output" "$GENIE_BASE/Log"
mkdir -p "$SCRIPTS_DIR"
mkdir -p "$UMLS_DIR/umls-normalizer-main/examples/input_genie_step3"
mkdir -p "$UMLS_DIR/umls-normalizer-main/examples/output_genie_step3"

# 3) Place required files
# - input JSONL at: $GENIE_BASE/Data/genie_train.jsonl   (or override INPUT_JSONL_REL/ABS)
# - UMLS dictionary file at UMLS_DICTIONARY_PATH
# - scripts A..J in $SCRIPTS_DIR (or override SCRIPT_DER_A ... SCRIPT_DER_J)

# 4) Optional: key for Step 9 (if OPENAI_API_KEY not exported at runtime)
# echo "sk-..." > "$GENIE_BASE/key.txt"
```

### Pre-run directory/file checks

```bash
test -d "$GENIE_BASE/Data" && test -d "$GENIE_BASE/Output" && test -d "$GENIE_BASE/Log"
test -d "$UMLS_DIR" && test -d "$UMLS_NORM_ROOT"
test -f "$GENIE_BASE/Data/genie_train.jsonl"
test -f "$UMLS_DICTIONARY_PATH"
test -f "$SCRIPTS_DIR/A_jsonl2table_0121.py"
test -f "$SCRIPTS_DIR/J_Result_evaluation_0121.py"
```

### Runtime and environment parameters

| Parameter | Default | Meaning / Effect |
|---|---|---|
| `PYVER` | `3.12` | Python version used when creating the conda env. |
| `ENV_BASE` | `/n/data1/hsph/biostat/celehs/lab/SHARE/From_Zebin/envs` | Base folder for CPU/GPU conda environments. |
| `ENV_CPU_DIR` | `${ENV_BASE}/genie_py${PYVER}_cpu_test` | Path to CPU environment. |
| `ENV_GPU_DIR` | `${ENV_BASE}/genie_py${PYVER}_gpu_cu128_test` | Path to GPU environment. |
| `USE_GPU` | `auto` | Global GPU mode selector: `auto`, `0`, or `1`. `auto` inspects `CUDA_VISIBLE_DEVICES`. |
| `UMLS_DIR` | `/n/data1/hsph/biostat/celehs/lab/SHARE/From_Zebin/` | Working directory used when adding UMLS normalizer to `PYTHONPATH`. |
| `UMLS_NORM_ROOT` | `/n/data1/hsph/biostat/celehs/lab/SHARE/From_Zebin/umls-normalizer-main` | Root added to `PYTHONPATH` for `umls_normalizer` imports. |
| `GENIE_BASE` | `/n/data1/hsph/biostat/celehs/lab/SHARE/From_Zebin/GENIE_Analysis_Consolidated_0121` | Main project root used for data/output/log defaults. |
| `GENIE_JSONL` | `${GENIE_BASE}/Data/genie_train.jsonl` | Informational export only (printed); actual Step 1 input is controlled by `INPUT_JSONL_REL/ABS`. |
| `UMLS_DICTIONARY_PATH` | `/n/data1/hsph/biostat/celehs/lab/SHARE/From_Zongxin/language-into-clinical-data/umls_dictionary.txt` | Dictionary passed to Step 3 UMLS normalization. Script exits if missing. |
| `KEY_FILE` | `${GENIE_BASE}/key.txt` | Fallback file to load `OPENAI_API_KEY` for Step 9 when key is not already in env. |
| `OPENAI_API_KEY` | (no default) | Required for Step 9 GPT scoring. If unset, pipeline reads from `KEY_FILE`. |

### UMLS normalization wiring (Step 3)

| Parameter | Default | Meaning / Effect |
|---|---|---|
| `CLR_DIR` | `umls_normalizer.cli` | Python module used as `python -m "$CLR_DIR"` for UMLS normalization. |
| `INPUT_UMLS_REL` | `umls-normalizer-main/examples/input_genie_step3/` | Input folder (relative to `UMLS_DIR`) where Step 2 CSV is copied for UMLS run. |
| `OUTPUT_UMLS_REL` | `umls-normalizer-main/examples/output_genie_step3/` | Output folder (relative to `UMLS_DIR`) read after UMLS run. |
| `UMLS_CSV_REL` | `genie_for_normalization.normalized.csv` | File name expected from UMLS output folder and used as downstream codebook source. |

### I/O path parameters across steps

| Parameter | Default | Meaning / Effect |
|---|---|---|
| `INPUT_JSONL_REL` | `Data/genie_train.jsonl` | Step 1 input path passed to script A relative to `GENIE_BASE`. |
| `INPUT_JSONL_ABS` | `${GENIE_BASE}/${INPUT_JSONL_REL}` | Absolute precheck path for Step 1 input JSONL. |
| `OUTPUT_CSV_REL` | `Data/genie_processed.csv` | Step 1 output and Step 2 input path (relative to `GENIE_BASE`). |
| `OUTPUT_CSV_ABS` | `${GENIE_BASE}/${OUTPUT_CSV_REL}` | Absolute precheck/log path for Step 1 output. |
| `NORMAL_CSV_REL` | `Data/genie_for_normalization.csv` | Step 2 output and Step 3 input path (relative to `GENIE_BASE`). |
| `NORMAL_CSV_ABS` | `${GENIE_BASE}/${NORMAL_CSV_REL}` | Absolute precheck/log path for Step 2 output. |
| `NORMALIZED_CSV_REL` | `Data/genie_normalized.csv` | Step 3 (C script) output and Step 4 input path (relative to `GENIE_BASE`). |
| `NORMALIZED_CSV_ABS` | `${GENIE_BASE}/${NORMALIZED_CSV_REL}` | Absolute precheck/log path for normalized CSV. |
| `DISCRET_CSV_REL` | `Data/genie_discretized.csv` | Step 4 output and Step 5 source path (relative to `GENIE_BASE`). |
| `DISCRET_CSV_ABS` | `${GENIE_BASE}/${DISCRET_CSV_REL}` | Absolute precheck/input path for Step 5. |
| `RUN_TAG` | `${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}` | Run identifier used for output subdirectory naming. |
| `PIPE_OUT_DIR` | `${OUT_DIR}/run_${RUN_TAG}` | Main per-run output directory for steps F-J artifacts. |
| `PRE6_OUT_DIR` | `${GENIE_BASE}/Data` | Folder used for Step 5 vocab output by default. |
| `SPPMI_NPZ_REL` | `Data/${DISCRET_STEM}_sppmi.npz` | Relative output path for Step 5 SPPMI matrix. |
| `SPPMI_NPZ_ABS` | `${GENIE_BASE}/${SPPMI_NPZ_REL}` | Absolute output path for Step 5 SPPMI matrix. |
| `E_VOCAB_ABS` | `${PRE6_OUT_DIR}/${DISCRET_STEM}_cui_vocab.csv` | Step 5 vocab output path and Step 6 vocab input path. |
| `F_OUT_DIR` | `${PIPE_OUT_DIR}/${DISCRET_STEM}_F_2Ddecomp` | Step 6 output directory. |
| `EMBED_CSV_ABS` | `${F_OUT_DIR}/embeddings.csv` | Optional Step 6 embeddings CSV output; also Step 7 input. |
| `G_PAIRS_ABS` | `${PIPE_OUT_DIR}/${DISCRET_STEM}_topk_pairs.csv` | Step 7 top-K pairs output path. |
| `CODEBOOK_CSV_ABS` | `${GENIE_BASE}/${UMLS_CSV_REL}` or `${GENIE_BASE}/Data/${UMLS_CSV_REL}` | Step 8 codebook input path. Default depends on whether `UMLS_CSV_REL` starts with `Data/`. |
| `H_PAIRS_ABS` | `${G_PAIRS_ABS}` | Step 8 pairs input path. |
| `H_OUT_ABS` | `${PIPE_OUT_DIR}/pairs_explained.csv` | Step 8 enriched pairs output path; Step 9 input. |
| `I_OUT_ABS` | `${PIPE_OUT_DIR}/gpt_scored.csv` | Step 9 scored output path; Step 10 input. |
| `I_AUDIT_ABS` | `${PIPE_OUT_DIR}/gpt_scored.audit.jsonl` | Step 9 audit JSONL output path. |
| `I_BATCHINPUT_ABS` | `${PIPE_OUT_DIR}/gpt_batchinput.jsonl` | Step 9 batch request JSONL output path (`GPT_MODE=batch`). |
| `J_OUT_ABS` | `${PIPE_OUT_DIR}/gpt_scored_labeled.csv` | Step 10 final labeled output path. |

### Script location overrides

| Parameter | Default | Meaning / Effect |
|---|---|---|
| `SCRIPTS_DIR` | `${GENIE_BASE}/Scripts_Refined_0121` | Base folder containing A-J Python scripts. |
| `SCRIPT_DER_A` | `${SCRIPTS_DIR}/A_jsonl2table_0121.py` | Script path override for Step 1. |
| `SCRIPT_DER_B` | `${SCRIPTS_DIR}/B_prepare4normalization_0121.py` | Script path override for Step 2. |
| `SCRIPT_DER_C` | `${SCRIPTS_DIR}/C_normalization_0121.py` | Script path override for Step 3C. |
| `SCRIPT_DER_D` | `${SCRIPTS_DIR}/D_discretization_0121.py` | Script path override for Step 4. |
| `SCRIPT_DER_E` | `${SCRIPTS_DIR}/E_load2tensor_0121.py` | Script path override for Step 5. |
| `SCRIPT_DER_F` | `${SCRIPTS_DIR}/F_2Ddecomp_GPU_0121.py` | Script path override for Step 6. |
| `SCRIPT_DER_G` | `${SCRIPTS_DIR}/G_Top_K_Pairs_0121.py` | Script path override for Step 7. |
| `SCRIPT_DER_H` | `${SCRIPTS_DIR}/H_Formality_Arrangement_0121.py` | Script path override for Step 8. |
| `SCRIPT_DER_I` | `${SCRIPTS_DIR}/I_GPT_scoring_0121.py` | Script path override for Step 9. |
| `SCRIPT_DER_J` | `${SCRIPTS_DIR}/J_Result_evaluation_0121.py` | Script path override for Step 10. |

### Step 4 parameter

| Parameter | Default | Meaning / Effect |
|---|---|---|
| `K_THRESHOLD` | `10` | Passed to script D as `--K` (minimum per-CUI threshold across statuses). |

### Step 5 (E_load2tensor) parameters

| Parameter | Default | Meaning / Effect |
|---|---|---|
| `E_USE_GPU` | `auto` | Step 5 device mode: `auto`, `0`, or `1`. `auto` enables GPU only if global GPU is enabled and CuPy is available. |
| `E_STATUS_FILTER` | `ALL` | Status filter passed to script E (comma-separated or `ALL`). |
| `E_PRESENT_ONLY` | `0` | If `1`, pipeline forces `E_STATUS_FILTER=PRESENT` and adds `--present_only`. |
| `E_TOKEN_DELIM` | `||` | Delimiter used in tokenization (`CUI||STATUS`). |
| `E_MIN_CUI_COUNT` | `1` | Minimum patient count per token. |
| `E_MIN_COOCCUR` | `1` | Minimum co-occurrence count per token pair. |
| `E_MAX_CUIS_PER_PATIENT` | `0` | Per-patient token cap in Step 5 (`0` means no cap). |
| `E_DTYPE` | `float32` | Output dtype for Step 5 PPMI values. |
| `E_PATIENT_COL` | `patient_id` | Patient ID column expected in Step 5 input CSV. |
| `E_CUI_COL` | `umls_top_cui` | CUI column expected in Step 5 input CSV. |
| `E_STATUS_COL` | `assertion_status_norm` | Status column expected in Step 5 input CSV. |

### Step 6 (F_2Ddecomp) parameters

| Parameter | Default | Meaning / Effect |
|---|---|---|
| `DECOMP_USE_GPU` | `auto` | Step 6 device mode: `auto`, `0`, or `1`. `auto` enables GPU only if global GPU is enabled and `torch.cuda` is available. |
| `DECOMP_K` | `128` | Number of embedding dimensions / eigenpairs. |
| `DECOMP_TOL` | `1e-3` | Solver convergence tolerance. |
| `DECOMP_MAXITER` | `2000` | Maximum iterations (used by `eigsh`). |
| `DECOMP_NCV` | `512` | Krylov subspace size for `eigsh`. |
| `DECOMP_WHICH` | `LA` | ARPACK selector for `eigsh` (e.g., `LA`, `LM`). |
| `DECOMP_LOBPCG_NITER` | `200` | Max iterations for `torch_lobpcg`. |
| `DECOMP_LOBPCG_INIT` | `randn` | LOBPCG initializer (`randn` or `rand`). |
| `DECOMP_SEED` | `0` | Random seed used for decomposition initialization. |
| `DECOMP_THREADS` | `${SLURM_CPUS_PER_TASK:-}` | Optional explicit CPU thread count passed to script F. |
| `DECOMP_SOLVER` | `torch_lobpcg` (GPU mode) or `eigsh` (CPU mode) | Solver override. If unset, default depends on `DECOMP_USE_GPU`. |
| `DECOMP_DEVICE` | `cuda` (GPU mode) or `cpu` (CPU mode) | Device string passed to script F. |
| `DECOMP_DTYPE` | `float32` (GPU mode) or `float64` (CPU mode) | Numeric dtype used in decomposition. |

### Step 7 (G_Top_K_Pairs) parameters

| Parameter | Default | Meaning / Effect |
|---|---|---|
| `G_K_SEARCH` | `2000` | Neighbors searched per query. |
| `G_TOP_PP` | `100` | Top pairs retained for `PRESENT~PRESENT`. |
| `G_TOP_OTHER` | `10` | Top pairs retained for all other status configs. |
| `G_USE_GPU` | `auto` | Step 7 FAISS mode: `auto`, `0`, or `1`. `auto` requires global GPU + FAISS GPU capability. |

### Step 9 (I_GPT_scoring) parameters

| Parameter | Default | Meaning / Effect |
|---|---|---|
| `GPT_MODEL` | `gpt-4.1-mini` | Model name passed to script I. |
| `GPT_MODE` | `sync` | Scoring mode: `sync` or `batch`. |
| `GPT_CONCURRENCY` | `10` | Concurrent requests in sync mode. |
| `GPT_MAX_RETRIES` | `5` | Retry limit in sync mode. |
| `GPT_POLL_SECONDS` | `10` | Polling interval in batch mode. |

### Step 10 (J_Result_evaluation) parameters

| Parameter | Default | Meaning / Effect |
|---|---|---|
| `J_MODE` | `tristate` | Evaluation mode passed to script J (`binary` or `tristate`). |
| `J_THRESHOLD` | `70` | Binary threshold (still passed even in tristate mode). |
| `J_LOW` | `50` | Lower cutoff for tristate mode. |
| `J_HIGH` | `85` | Upper cutoff for tristate mode. |

### Environment variables read from scheduler/runtime

| Variable | Source | Meaning / Effect |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | Slurm/runtime | Used by `USE_GPU=auto` detection and GPU log suffixes. |
| `SLURM_JOB_ID` | Slurm | Used as first-choice default for `RUN_TAG`. |
| `SLURM_CPUS_PER_TASK` | Slurm | Used as default for `DECOMP_THREADS`. |
| `SLURM_NODELIST` | Slurm | Printed in sanity logs (informational). |

---

## A. `A_jsonl2table_0121.py`

### Tuning parameters

| Parameter | Default | Meaning / Effect |
|---|---|---|
| `--key-delim` | `-` | Delimiter for the `key` field (expected format: `patient-admission-message`). If the key does not split into exactly 3 parts, the parsed IDs are set to `None`. |

### Schema/Path parameters

| Parameter | Default | Meaning / Effect |
|---|---|---|
| `--base-dir` | `.` | Base directory for relative paths. |
| `--input-jsonl` | `Data/genie_train.jsonl` | Input GENIE jsonl. |
| `--output-csv` | `Data/genie_processed.csv` | Output CSV. |

---

## B. `B_prepare4normalization_0121.py`

### Tuning parameters

None. (All parameters are schema/path.)

### Schema/Path parameters

| Parameter | Default | Meaning / Effect |
|---|---|---|
| `--base-dir` | `.` | Base directory for relative paths. |
| `--input-csv` | `Data/genie_processed.csv` | Input CSV from A. |
| `--output-csv` | `Data/vector_for_normalization.csv` | Output CSV for external UMLS normalization. |

---

## C. `C_normalization_0121.py`

### Tuning parameters

| Parameter | Default | Meaning / Effect |
|---|---|---|
| `--align` | `positional` | Alignment mode between the full data and normalization output. `positional` matches rows by order; `id` merges on an ID column. |

### Schema/Path parameters

| Parameter | Default | Meaning / Effect |
|---|---|---|
| `--base-dir` | `.` | Base directory for relative paths. |
| `--input-norm` | `Data/vector_normalized.csv` | UMLS-normalized CSV (must contain `umls_top_cui`). |
| `--input-full` | `Data/genie_processed.csv` | Full CSV from A. |
| `--output-csv` | `Data/genie_normalized_1216.csv` | Output normalized CSV. |
| `--id-col` | `id` | ID column used when `--align id`. If missing in the full CSV, it is generated as 1..N. |

---

## D. `D_discretization_0121.py`

### Tuning parameters

| Parameter | Default | Meaning / Effect |
|---|---|---|
| `--K` | `10` | Minimum count threshold per CUI **across statuses**. A CUI is kept if its **max** count over statuses is ≥ K. |

### Schema/Path parameters

| Parameter | Default | Meaning / Effect |
|---|---|---|
| `--base-dir` | `.` | Base directory for relative paths. |
| `--input-csv` | `Data/genie_normalized_1216.csv` | Input CSV from C. |
| `--output-csv` | `Data/genie_discretized_1216.csv` | Output discretized CSV. |

---

## E. `E_load2tensor_0121.py`

### Tuning parameters

| Parameter | Default | Meaning / Effect |
|---|---|---|
| `--status_filter` | `ALL` | Comma-separated statuses to keep (e.g., `PRESENT,HISTORY`). `ALL` keeps all statuses. |
| `--present_only` | `false` | Convenience flag to restrict to `PRESENT` only (overrides `--status_filter`). |
| `--token_delim` | `||` | Delimiter for CUI~STATUS tokens (token = `CUI||STATUS`). |
| `--min_cui_count` | `1` | Minimum **patient** count per token to keep. |
| `--min_cooccur` | `1` | Minimum co-occurrence count for a token pair to be retained. |
| `--max_cuis_per_patient` | `0` | Cap the number of tokens per patient (0 = no cap). |
| `--dtype` | `float32` | Output dtype for PPMI values. |
| `--device` | `cpu` | `cpu` or `cuda` (CUDA requires CuPy). |

### Schema/Path parameters

| Parameter | Default | Meaning / Effect |
|---|---|---|
| `--input_csv` | (required) | Discretized CSV (must include patient, CUI, status columns). |
| `--output_npz` | (required) | Output SPPMI sparse matrix (`.npz`). |
| `--vocab_csv` | `` | Vocab CSV path (optional; defaults to `output_npz + .vocab.csv`). |
| `--patient_col` | `patient_id` | Patient ID column name. |
| `--cui_col` | `umls_top_cui` | CUI column name. |
| `--status_col` | `assertion_status_norm` | Assertion status column name. |

---

## F. `F_2Ddecomp_GPU_0121.py`

### Tuning parameters

| Parameter | Default | Meaning / Effect |
|---|---|---|
| `--threshold` | `None` | Drop entries with value `< threshold` before decomposition. |
| `--drop_diagonal` | `false` | Zero out diagonal entries (self‑co‑occurrence). |
| `--symmetrize` | `false` | Replace A with `(A + A.T)/2` for symmetry. |
| `--dtype` | `float64` | Compute dtype (`float64` for stability; `float32` for speed/GPU). |
| `--solver` | `eigsh` | `eigsh` (CPU ARPACK) or `torch_lobpcg` (CPU/GPU via PyTorch). |
| `--device` | `cpu` | Device for `torch_lobpcg` (`cpu`, `cuda`, `cuda:0`, etc.). |
| `--k` | `128` | Number of eigenpairs to compute. |
| `--which` | `LA` | ARPACK selector for `eigsh` (e.g., `LA`, `LM`, `SA`). |
| `--tol` | `1e-3` | Convergence tolerance for `eigsh` / `lobpcg`. |
| `--maxiter` | `2000` | Max iterations for `eigsh`. |
| `--ncv` | `None` | Krylov subspace size for `eigsh`. |
| `--lobpcg_niter` | `200` | Max iterations for `torch.lobpcg`. |
| `--lobpcg_init` | `randn` | Initialization for LOBPCG (`randn` or `rand`). |
| `--seed` | `0` | Random seed for LOBPCG initialization. |
| `--embed` | `sqrt_evals` | Embedding construction: `evecs` or `sqrt_evals`. |
| `--threads` | `None` | Set CPU threads for numpy/scipy/torch (best-effort). |

### Schema/Path parameters

| Parameter | Default | Meaning / Effect |
|---|---|---|
| `--input_npz` | (required) | Input SPPMI `.npz`. |
| `--out_dir` | (required) | Output directory for embeddings and metadata. |
| `--save_A` | `false` | Save processed matrix as `A_csr.npz` in `out_dir`. |
| `--vocab_csv` | `` | Vocab CSV from E (required if writing embeddings CSV). |
| `--output_csv` | `` | Optional embeddings CSV path. |
| `--emb_prefix` | `emb_` | Prefix for embedding columns in CSV. |
| `--cui_col` | `cui` | Column in vocab CSV used as the ID column. |
| `--out_cui_col` | `umls_top_cui` | Column name for IDs in embeddings CSV. |

---

## G. `G_Top_K_Pairs_0121.py`

### Tuning parameters

| Parameter | Default | Meaning / Effect |
|---|---|---|
| `--k_search` | `2000` | Neighbors per query for each ordered status‑pair search. |
| `--top_pp` | `100` | Number of pairs to keep for `PRESENT~PRESENT` (or `pp_label~pp_label`). |
| `--top_other` | `10` | Number of pairs to keep for all other status configurations. |
| `--pp_label` | `PRESENT` | Status label treated as the “pp” bucket for `top_pp`. |
| `--require_faiss` | `false` | If set, error out when FAISS is unavailable. |
| `--faiss-gpu` | `false` | Use FAISS GPU index (requires faiss‑gpu + CUDA). |

### Schema/Path parameters

| Parameter | Default | Meaning / Effect |
|---|---|---|
| `--input_csv` | (required) | Input embeddings CSV. |
| `--output_csv` | (required) | Output pairs CSV. |
| `--pair_token_col` | `pair_token` | Column containing `CUI||STATUS` tokens. |
| `--cui_col` | `umls_top_cui` | CUI column (if `pair_token_col` is absent). |
| `--status_col` | `` | Optional status column (if absent, `default_status` is used). |
| `--default_status` | `PRESENT` | Status assigned when no status column is provided. |
| `--emb_prefix` | `emb_` | Prefix for embedding columns. |
| `--delim` | `||` | Delimiter for `pair_token` parsing. |
| `--sep` | `,` | CSV separator. |
| `--auto_sep` | `false` | Auto-detect separator (CSV/TSV). |

---

## H. `H_Formality_Arrangement_0121.py`

### Tuning parameters

None. (All parameters are schema/path.)

### Schema/Path parameters

| Parameter | Default | Meaning / Effect |
|---|---|---|
| `--base-dir` | `.` | Base directory for relative paths. |
| `--pairs-csv` | (required) | Pairs CSV (from G). |
| `--codebook-csv` | (required) | Codebook CSV with `umls_top_*` columns. |
| `--out-csv` | (required) | Output enriched pairs CSV. |

---

## I. `I_GPT_scoring_0121.py`

### Tuning parameters

| Parameter | Default | Meaning / Effect |
|---|---|---|
| `--model` | (required) | Model name used for scoring. |
| `--mode` | (required) | `sync` or `batch`. |
| `--concurrency` | `10` | Concurrent requests in `sync` mode. |
| `--max-retries` | `5` | Max retries per request in `sync` mode. |
| `--poll-seconds` | `10` | Polling interval for `batch` mode. |
| `--assertion-optional` | `false` | If set, missing assertion columns default to `unknown` instead of erroring. |

### Schema/Path parameters

| Parameter | Default | Meaning / Effect |
|---|---|---|
| `--input` | (required) | Input CSV with pair columns. |
| `--output` | (required) | Output CSV with scores. |
| `--batchinput` | `batchinput.jsonl` | Batch input JSONL path (batch mode). |
| `--audit-jsonl` | `` | Optional audit JSONL (full inputs/requests/responses). |
| `--id-col` | `pair_id` | Pair ID column. |
| `--a-cui-col` | `a_cui` | Concept A CUI column. |
| `--a-term-col` | `a_term` | Concept A term column. |
| `--a-semantic-type-col` | `a_semantic_type` | Concept A semantic type column. |
| `--a-assertion-col` | `a_assertion_status` | Concept A assertion status column. |
| `--b-cui-col` | `b_cui` | Concept B CUI column. |
| `--b-term-col` | `b_term` | Concept B term column. |
| `--b-semantic-type-col` | `b_semantic_type` | Concept B semantic type column. |
| `--b-assertion-col` | `b_assertion_status` | Concept B assertion status column. |

Note: `OPENAI_API_KEY` must be set in the environment (or provided via the pipeline’s key file) before running this script.

---

## J. `J_Result_evaluation_0121.py`

### Tuning parameters

| Parameter | Default | Meaning / Effect |
|---|---|---|
| `--mode` | `tristate` | `binary` (single cutoff) or `tristate` (low/high cutoffs). |
| `--threshold` | `None` | Binary cutoff: score ≥ threshold ⇒ positive (used in `binary` mode). |
| `--low` | `50.0` | Tri‑state low cutoff: score ≤ low ⇒ negative. |
| `--high` | `85.0` | Tri‑state high cutoff: score ≥ high ⇒ positive. |
| `--chunksize` | `None` | Process file in chunks to reduce memory use. |

### Schema/Path parameters

| Parameter | Default | Meaning / Effect |
|---|---|---|
| `--input` | (required) | Input CSV/TSV path. |
| `--output` | (required) | Output CSV/TSV path. |
| `--sep` | `None` | Override delimiter (defaults based on file extension). |
| `--score_col` | `relationship_score` | Score column name. |
| `--conf_col` | `confidence` | Confidence column name (optional). |
