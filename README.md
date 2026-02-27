# GENIE A–J Pipeline (Portable Scripts)

This repo contains portable, CLI-driven scripts for the GENIE A–J pipeline. Each script accepts paths (so it can run from any working directory) and a small set of tuning parameters that control filtering, decomposition, and scoring behavior.

## Pipeline Overview

A. `A_jsonl2table_0121.py` — Flatten GENIE jsonl into a CSV table.

B. `B_prepare4normalization_0121.py` — Build a minimal table for external UMLS normalization (entity + note).

C. `C_normalization_0121.py` — Merge UMLS normalization output back into the full table.

D. `D_discretization_0121.py` — Filter + discretize assertion statuses into a controlled set.

E. `E_load2tensor_0121.py` — Build a sparse SPPMI matrix on CUI||STATUS tokens (with optional PRESENT-only mode).

F. `F_2Ddecomp_GPU_0121.py` — Decompose the SPPMI matrix into embeddings (CPU or GPU).

G. `G_Top_K_Pairs_0121.py` — Find top cosine-similar CUI pairs by status configuration.

H. `H_Formality_Arrangement_0121.py` — Enrich pairs with terms + semantic types.

I. `I_GPT_scoring_0121.py` — Score concept pairs with an LLM (sync or batch).

J. `J_Result_evaluation_0121.py` — Apply score cutoffs to label results.

## Inputs and Outputs (High-Level)

- A: `*.jsonl` → `Data/genie_processed.csv`
- B: `Data/genie_processed.csv` → `Data/vector_for_normalization.csv`
- C: `Data/vector_normalized.csv` + `Data/genie_processed.csv` → `Data/genie_normalized.csv`
- D: `Data/genie_normalized.csv` → `Data/genie_discretized.csv`
- E: `Data/genie_discretized.csv` → `Data/*_sppmi.npz` + vocab CSV
- F: `Data/*_sppmi.npz` → `Output/*/embeddings.npy` (+ optional embeddings CSV)
- G: embeddings CSV → top‑K pairs CSV
- H: top‑K pairs CSV + codebook CSV → enriched pairs CSV
- I: enriched pairs CSV → LLM-scored CSV (+ optional audit JSONL)
- J: scored CSV → labeled CSV

## Note on CUI||STATUS Tokens (Step E)

Step E builds the SPPMI matrix on **CUI||STATUS** tokens by default (delimiter configurable). This keeps assertion status in downstream embeddings and top‑K pairing. If you want to consider only PRESENT concepts, enable the PRESENT-only option in Step E.

## Future Improvement (Longitudinal Mode)

Current co-occurrence in Step E is patient-level and non-temporal. A future longitudinal mode could be added for datasets with richer time structure (for example, encounter-level windows or time-bounded co-occurrence). This would require preserving time/encounter metadata through Steps C and D and extending Step E to accept time-aware grouping/window arguments.

## Documentation

- `CODEBOOK.md` contains the full tuning-parameter codebook for scripts A–J.
