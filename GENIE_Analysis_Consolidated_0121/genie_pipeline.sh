#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 20
#SBATCH --mem=80G
#SBATCH -t 0-08:00
#SBATCH -p short
#SBATCH -o GENIE_CPU_0206_%j.out
#SBATCH -e GENIE_CPU_0206_%j.err

# ----------------------------
# A–J Pipeline (Slurm)
# - Creates/uses a conda env
# - Clones umls-normalizer if missing
# - Patches local scripts to be cluster-runnable (base path + example blocks)
# - Runs A..J in order
# - Records wall time, CPU%, Max RSS, and (if GPU present) GPU util/mem
# ----------------------------


# ========= Environment Setup (CPU/GPU adaptive via sbatch overrides) =========
# Usage:
#   CPU (default headers):  sbatch yourjob.sh
#   GPU (override headers): sbatch -p gpu --gres=gpu:1 -t 0-00:15 yourjob.sh
#
# Optional env vars (override defaults):
#   PYVER=3.12
#   ENV_BASE=/n/data1/hsph/biostat/celehs/lab/zew324/envs
#   ENV_CPU_DIR=$ENV_BASE/genie_py3.12_cpu
#   ENV_GPU_DIR=$ENV_BASE/genie_py3.12_gpu_cu128
#   USE_GPU=auto|0|1   (auto detects GPU allocation; 0/1 forces)
#   UMLS_DIR=...
#   UMLS_NORM_ROOT=...

set -euo pipefail

# ========= Load conda =========
set +u
module load conda/miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
set -u

# ========= Paths / defaults =========
PYVER="${PYVER:-3.12}"

ENV_BASE="${ENV_BASE:-/n/data1/hsph/biostat/celehs/lab/SHARE/From_Zebin/envs}"
# ENV_CPU_DIR="${ENV_CPU_DIR:-${ENV_BASE}/genie_py${PYVER}_cpu}"
# ENV_GPU_DIR="${ENV_GPU_DIR:-${ENV_BASE}/genie_py${PYVER}_gpu_cu128}"
ENV_CPU_DIR="${ENV_CPU_DIR:-${ENV_BASE}/genie_py${PYVER}_cpu_test}"
ENV_GPU_DIR="${ENV_GPU_DIR:-${ENV_BASE}/genie_py${PYVER}_gpu_cu128_test}"

USE_GPU="${USE_GPU:-auto}"

UMLS_DIR="${UMLS_DIR:-/n/data1/hsph/biostat/celehs/lab/SHARE/From_Zebin/}"
UMLS_NORM_ROOT="${UMLS_NORM_ROOT:-/n/data1/hsph/biostat/celehs/lab/SHARE/From_Zebin/umls-normalizer-main}"
UMLS_DIR="${UMLS_DIR%/}"
UMLS_NORM_ROOT="${UMLS_NORM_ROOT%/}"
export UMLS_DIR UMLS_NORM_ROOT

# ========= Detect whether SLURM allocated GPU =========
if [[ "${USE_GPU}" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 \
     && [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] \
     && [[ "${CUDA_VISIBLE_DEVICES}" != "-1" ]]; then
    WANT_GPU=1
  else
    WANT_GPU=0
  fi
else
  WANT_GPU="${USE_GPU}"
fi

ENV_DIR="$([[ "${WANT_GPU}" == "1" ]] && echo "${ENV_GPU_DIR}" || echo "${ENV_CPU_DIR}")"
echo "=== Environment selection ==="
echo "WANT_GPU=${WANT_GPU} (USE_GPU=${USE_GPU}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
echo "ENV_DIR=${ENV_DIR}"
echo

# ========= Package requirements =========
# Common pins (NOTE: we will patch numpy/scipy in-place depending on WANT_GPU)
COMMON_PKGS=(
  "numpy==1.26.4"
  "pandas==2.3.3"
  "tqdm==4.67.1"
  "psutil>=5.9"
  "openai>=1.0.0"
  "transformers==4.57.3"
  "scipy<1.15"
)

# --- Patch COMMON_PKGS for GPU mode (do not rename variables) ---
if [[ "${WANT_GPU}" == "1" ]]; then
  # faiss-gpu-cu12 requires NumPy 2.x (your pip check showed numpy>=2,<3 requirement)
  COMMON_PKGS[0]="numpy>=2,<3"
  # ensure SciPy supports NumPy 2; SciPy 1.13+ is NumPy 2 compatible :contentReference[oaicite:3]{index=3}
  for i in "${!COMMON_PKGS[@]}"; do
    if [[ "${COMMON_PKGS[$i]}" == scipy* ]]; then
      COMMON_PKGS[$i]="scipy>=1.13,<1.15"
    fi
  done
fi

# ========= Environment check / setup =========
env_check() {
  local reqs
  reqs="$(printf "%s\n" "${COMMON_PKGS[@]}")"

  REQUIREMENTS="$reqs" PYVER="$PYVER" WANT_GPU="$WANT_GPU" python - <<'PY'
import os
import sys
import importlib.metadata as md

errors = []

# Python version
expected = os.environ.get("PYVER", "")
pyver = f"{sys.version_info[0]}.{sys.version_info[1]}"
if expected and pyver != expected:
    errors.append(f"python {pyver} (expected {expected})")

# Requirement checks
try:
    from packaging.requirements import Requirement
    from packaging.version import Version
    have_packaging = True
except Exception as e:
    errors.append(f"packaging not available ({type(e).__name__}: {e})")
    have_packaging = False

if have_packaging:
    reqs = [line.strip() for line in os.environ.get("REQUIREMENTS", "").splitlines() if line.strip()]
    for req_str in reqs:
        try:
            req = Requirement(req_str)
        except Exception as e:
            errors.append(f"bad requirement '{req_str}' ({type(e).__name__}: {e})")
            continue
        try:
            ver = md.version(req.name)
        except Exception as e:
            errors.append(f"missing {req.name} ({type(e).__name__}: {e})")
            continue
        if req.specifier and ver not in req.specifier:
            errors.append(f"{req.name}=={ver} does not satisfy {req.specifier}")

want_gpu = os.environ.get("WANT_GPU", "0") == "1"

# Torch
try:
    import torch
except Exception as e:
    errors.append(f"torch import failed ({type(e).__name__}: {e})")
    torch = None

if want_gpu:
    # Torch CUDA build
    if torch is not None:
        cuda_ver = getattr(torch.version, "cuda", None)
        if not cuda_ver:
            errors.append("torch is CPU build (torch.version.cuda is None)")
        elif not str(cuda_ver).startswith("12.8"):
            errors.append(f"torch CUDA version is {cuda_ver}, expected 12.8.x")

    # CUDA runtime wheels
    try:
        v = md.version("nvidia-cuda-runtime-cu12")
        if have_packaging and Version(v) < Version("12.8"):
            errors.append(f"nvidia-cuda-runtime-cu12 {v} < 12.8")
    except Exception as e:
        errors.append(f"missing nvidia-cuda-runtime-cu12 ({type(e).__name__}: {e})")

    # faiss-gpu build
    try:
        import faiss  # type: ignore
        if not hasattr(faiss, "get_num_gpus"):
            errors.append("faiss is CPU build (no get_num_gpus)")
    except Exception as e:
        errors.append(f"faiss import failed ({type(e).__name__}: {e})")

    # cupy
    try:
        import cupy  # type: ignore
    except Exception as e:
        errors.append(f"cupy import failed ({type(e).__name__}: {e})")
else:
    # CPU-only expectations
    if torch is not None:
        if getattr(torch.version, "cuda", None) is not None:
            errors.append(f"torch is CUDA build ({torch.version.cuda}) in CPU env")

    try:
        import faiss  # type: ignore
        if hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0:
            errors.append("faiss is GPU build in CPU env")
    except Exception as e:
        errors.append(f"faiss import failed ({type(e).__name__}: {e})")

    # GPU-only packages should not be present in CPU env
    for pkg in [
        "faiss-gpu-cu12",
        "cupy-cuda12x",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cuda-nvrtc-cu12",
        "nvidia-cuda-cupti-cu12",
        "nvidia-nvjitlink-cu12",
        "nvidia-nvtx-cu12",
    ]:
        try:
            md.version(pkg)
            errors.append(f"{pkg} is installed in CPU env")
        except md.PackageNotFoundError:
            pass
        except Exception as e:
            errors.append(f"error checking {pkg} ({type(e).__name__}: {e})")

if errors:
    print("ENV_CHECK_FAILED")
    for e in errors:
        print(" -", e)
    raise SystemExit(1)

print("ENV_CHECK_OK")
PY
}

ENV_NEEDS_SETUP=0

if [[ -d "${ENV_DIR}" ]]; then
  set +u
  conda activate "${ENV_DIR}"
  set -u
  if env_check; then
    echo "=== Environment check passed; using existing env ==="
  else
    echo "=== Environment check failed; resetting env ==="
    ENV_NEEDS_SETUP=1
    set +u
    conda deactivate || true
    set -u
  fi
else
  ENV_NEEDS_SETUP=1
fi

if [[ "${ENV_NEEDS_SETUP}" == "1" ]]; then
  if [[ -d "${ENV_DIR}" ]]; then
    echo "=== Removing conda env: ${ENV_DIR} ==="
    set +u
    conda env remove -p "${ENV_DIR}" -y || conda remove -p "${ENV_DIR}" --all -y
    set -u
  fi

  echo "=== Creating conda env: ${ENV_DIR} (python=${PYVER}) ==="
  set +u
  conda create -y -p "${ENV_DIR}" "python=${PYVER}"
  set -u

  # Activate after creation
  set +u
  conda activate "${ENV_DIR}"
  set -u

  echo "=== Ensuring pip is up to date ==="
  python -m pip install -U pip

  echo "=== Installing/ensuring core packages ==="
  python -m pip install -U "${COMMON_PKGS[@]}"

  # Torch + Faiss + CuPy differ by mode.
  # IMPORTANT: faiss-cpu and faiss-gpu cannot coexist in one env; we uninstall any existing faiss variant first.
  python -m pip uninstall -y faiss-cpu faiss-gpu faiss-gpu-cu12 >/dev/null 2>&1 || true

  if [[ "${WANT_GPU}" == "1" ]]; then
    echo "=== Installing GPU stack (torch cu128, faiss-gpu, cupy) ==="

  # 1) Prevent stale CUDA runtime wheels from lingering (this is what bit you: 12.1.105)
    #    Remove *all* nvidia-*-cu12 wheels, then reinstall coherently.
    python -m pip freeze | awk -F== '/^nvidia-.*-cu12==/ {print $1}' | xargs -r python -m pip uninstall -y
    python -m pip cache purge >/dev/null 2>&1 || true

    # 1b) Pin CUDA runtime wheels to 12.8.* so libcudart exports cudaGetDriverEntryPointByVersion
    #     (This avoids the 12.1.105 trap that breaks torch import.)
    python -m pip install -U --no-cache-dir --force-reinstall \
      nvidia-cuda-runtime-cu12==12.8.90 \
      nvidia-cuda-nvrtc-cu12==12.8.93 \
      nvidia-cuda-cupti-cu12==12.8.90 \
      nvidia-nvjitlink-cu12==12.8.93 \
      nvidia-nvtx-cu12==12.8.90

    # 2) Install torch cu128 (force reinstall so it cannot keep an older dependency set)
    python -m pip install -U --no-cache-dir --upgrade-strategy eager --force-reinstall \
      --index-url https://download.pytorch.org/whl/cu128 \
      --extra-index-url https://pypi.org/simple \
      torch==2.9.1

    # 3) Hard guard: libcudart must be new enough (>= 12.8.* is required here)
    python - <<'PY'
import importlib.metadata as md
v = md.version("nvidia-cuda-runtime-cu12")
print("nvidia-cuda-runtime-cu12:", v)
maj, minor, *_ = map(int, v.split("."))
if (maj, minor) < (12, 8):
    raise SystemExit(f"ERROR: CUDA runtime too old ({v}); need >= 12.8.* for cu128 torch.")
PY

    # 4) Sanity import torch NOW (so you catch breakage immediately)
    python -c "import torch; print('torch import OK'); print(torch.__version__, torch.version.cuda)"

    # 5) Install faiss/cupy WITHOUT letting them rewrite CUDA runtime wheels
    #    Then add CuPy's required dep fastrlock explicitly (because we used --no-deps). :contentReference[oaicite:4]{index=4}
    python -m pip install -U --no-cache-dir --no-deps faiss-gpu-cu12 cupy-cuda12x
    python -m pip install -U fastrlock

    # Optional: detect conflicts early
    python -m pip check

  fi

  if [[ "${WANT_GPU}" == "0" ]]; then
    echo "=== Installing CPU stack (torch cpu, faiss-cpu) ==="

    # Keep CPU env clean: remove GPU-only packages if they ever got installed
    python -m pip freeze | awk -F== '/^nvidia-.*-cu12==/ {print $1}' | xargs -r python -m pip uninstall -y
    python -m pip uninstall -y cupy-cuda12x fastrlock faiss-gpu-cu12 >/dev/null 2>&1 || true
    python -m pip cache purge >/dev/null 2>&1 || true

    python -m pip install -U --no-cache-dir --upgrade-strategy eager --force-reinstall \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple \
      torch==2.9.1

    python -m pip install -U "faiss-cpu==1.13.1"

    # Optional: detect conflicts early
    python -m pip check

    # Sanity
    python -c "import torch; print('torch import OK'); print(torch.__version__, getattr(torch.version,'cuda',None))"
  fi
fi


# ========= Ensure CUDA wheel libs are visible to dynamic loader =========
# CuPy may require libnvrtc.so.12 at runtime. With pip wheels, that library can
# live under site-packages/nvidia/*/lib and may not be on LD_LIBRARY_PATH.
CUDA_WHEEL_LIBS="$(
python - <<'PY'
import glob
import os
import site

paths = []
site_dirs = []
try:
    site_dirs.extend(site.getsitepackages())
except Exception:
    pass
try:
    u = site.getusersitepackages()
    if u:
        site_dirs.append(u)
except Exception:
    pass

for base in site_dirs:
    for p in glob.glob(os.path.join(base, "nvidia", "*", "lib")):
        if os.path.isdir(p):
            paths.append(os.path.realpath(p))

# De-duplicate while preserving order
seen = set()
uniq = []
for p in paths:
    if p not in seen:
        seen.add(p)
        uniq.append(p)

print(":".join(uniq))
PY
)"

if [[ -n "${CUDA_WHEEL_LIBS}" ]]; then
  export LD_LIBRARY_PATH="${CUDA_WHEEL_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  echo "=== CUDA wheel library paths added to LD_LIBRARY_PATH ==="
  echo "$CUDA_WHEEL_LIBS"
fi


# ========= UMLS normalizer on PYTHONPATH =========
set +u
cd "${UMLS_DIR}"
export PYTHONPATH="${UMLS_NORM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
set -u

python - << 'EOF'
import umls_normalizer
print("Loaded umls_normalizer from:", umls_normalizer.__file__)
EOF

echo
echo "=== Final sanity checks ==="
python - << 'EOF'
import os, sys, platform
import numpy, pandas, tqdm, psutil, torch, transformers, scipy

# Optional deps
faiss = None
faiss_err = None
try:
    import faiss  # type: ignore
except Exception as e:
    faiss_err = e

cupy = None
cupy_err = None
try:
    import cupy  # type: ignore
except Exception as e:
    cupy_err = e

openai = None
openai_err = None
try:
    import openai  # type: ignore
except Exception as e:
    openai_err = e

# Mode detection
cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
torch_gpu = False
torch_gpu_count = 0
torch_gpu_name = None
try:
    torch_gpu = bool(torch.cuda.is_available())
    if torch_gpu:
        torch_gpu_count = int(torch.cuda.device_count())
        if torch_gpu_count > 0:
            torch_gpu_name = torch.cuda.get_device_name(0)
except Exception:
    pass

faiss_gpu_capable = False
faiss_gpu_count = None
if faiss is not None and hasattr(faiss, "get_num_gpus"):
    faiss_gpu_capable = True
    try:
        faiss_gpu_count = int(faiss.get_num_gpus())
    except Exception:
        pass

cupy_gpu = False
cupy_gpu_count = None
if cupy is not None:
    try:
        cupy_gpu_count = int(cupy.cuda.runtime.getDeviceCount())
        cupy_gpu = (cupy_gpu_count > 0)
    except Exception:
        pass

has_allocation_hint = (cuda_visible is not None and cuda_visible != "" and cuda_visible != "-1")
mode = "GPU" if (has_allocation_hint and (torch_gpu or cupy_gpu or (faiss_gpu_count and faiss_gpu_count > 0) or faiss_gpu_capable)) else "CPU"

print("=== Mode detection ===")
print("MODE:", mode)
print("CUDA_VISIBLE_DEVICES:", cuda_visible)

print("\n=== Python / Platform ===")
print("python:", sys.version.replace("\n", " "))
print("executable:", sys.executable)
print("platform:", platform.platform())
print("machine:", platform.machine())
print("SLURM_JOB_ID:", os.environ.get("SLURM_JOB_ID"))
print("SLURM_NODELIST:", os.environ.get("SLURM_NODELIST"))

print("\n=== UMLS normalizer ===")
try:
    import umls_normalizer
    print("umls_normalizer:", umls_normalizer.__file__)
except Exception as e:
    print("umls_normalizer: NOT AVAILABLE", f"({type(e).__name__}: {e})")

print("\n=== Core packages ===")
print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("tqdm:", tqdm.__version__)
print("psutil:", psutil.__version__)
print("transformers:", transformers.__version__)
print("scipy:", scipy.__version__)

print("\n=== Torch ===")
print("torch:", torch.__version__)
print("torch.version.cuda:", getattr(torch.version, "cuda", None))
print("torch.cuda.is_available:", torch_gpu)
print("torch.cuda.device_count:", torch_gpu_count)
if torch_gpu_name is not None:
    print("torch.cuda.device0:", torch_gpu_name)

print("\n=== openai ===")
if openai is None:
    print("openai: NOT AVAILABLE", f"({type(openai_err).__name__}: {openai_err})")
else:
    print("openai:", getattr(openai, "__version__", "unknown"))

print("\n=== faiss ===")
if faiss is None:
    print("faiss: NOT AVAILABLE", f"({type(faiss_err).__name__}: {faiss_err})")
else:
    print("faiss:", getattr(faiss, "__version__", "unknown"))
    if hasattr(faiss, "get_num_gpus"):
        print("faiss build:", "GPU-capable (has get_num_gpus)")
        print("faiss.get_num_gpus:", faiss_gpu_count)
    else:
        print("faiss build:", "CPU-only (no get_num_gpus)")

print("\n=== cupy ===")
if cupy is None:
    print("cupy: NOT AVAILABLE", f"({type(cupy_err).__name__}: {cupy_err})")
else:
    print("cupy:", cupy.__version__)
    try:
        v = cupy.cuda.runtime.getVersion()
        print("cupy CUDA runtime:", f"{v//1000}.{(v%1000)//10} (raw={v})")
    except Exception as e:
        print("cupy runtime probe: ERROR", repr(e))
    try:
        print("cupy device_count:", cupy_gpu_count)
    except Exception as e:
        print("cupy device probe: ERROR", repr(e))
EOF

echo
echo "=== loaded modules ==="
module -t list 2>&1 || true

echo "=== nvidia-smi (if available) ==="
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "nvidia-smi: not found"



# ========= User-configurable =========
export GENIE_BASE="${GENIE_BASE:-/n/data1/hsph/biostat/celehs/lab/SHARE/From_Zebin/GENIE_Analysis_Consolidated_0121}"
export GENIE_JSONL="${GENIE_JSONL:-$GENIE_BASE/Data/genie_train.jsonl}"
export UMLS_DICTIONARY_PATH="${UMLS_DICTIONARY_PATH:-/n/data1/hsph/biostat/celehs/lab/SHARE/From_Zongxin/language-into-clinical-data/umls_dictionary.txt}"
export USE_GPU="${USE_GPU:-0}"   # submit with GPU via: sbatch --gres=gpu:1 -p gpu ...

DATA_DIR="$GENIE_BASE/Data"
OUT_DIR="$GENIE_BASE/Output"
LOG_DIR="$GENIE_BASE/Log"

mkdir -p "$DATA_DIR" "$OUT_DIR" "$LOG_DIR"

echo "GENIE_BASE=$GENIE_BASE"
echo "GENIE_JSONL=$GENIE_JSONL"
echo "UMLS_DICTIONARY_PATH=$UMLS_DICTIONARY_PATH"
echo "USE_GPU=$WANT_GPU"
echo

if [[ ! -f "$UMLS_DICTIONARY_PATH" ]]; then
  echo "ERROR: UMLS_DICTIONARY_PATH does not exist: $UMLS_DICTIONARY_PATH"
  exit 2
fi

# ========= Timing and Resource Recording =========
# Time stamp

run_timed() {
  local name="$1"; shift
  if [[ "$#" -eq 0 ]]; then
    echo "ERROR: run_timed(${name}) received no command"
    return 2
  fi
  local cmd_display
  cmd_display="$(printf '%q ' "$@")"
  local ts="$(date +%Y%m%d)"
  local time_log="${LOG_DIR}/${name}_${ts}.time.txt"
  local out_log="${LOG_DIR}/${name}_${ts}.stdout.txt"
  local err_log="${LOG_DIR}/${name}_${ts}.stderr.txt"

  echo "[$(date)] TIMED RUN (${name}): ${cmd_display% }"
  /usr/bin/time -v -o "$time_log" env -u SHELLOPTS "$@" >"$out_log" 2>"$err_log"
  echo "[$(date)] TIME summary: $time_log"
}

gpu_monitor_start() {
  local out_csv="$1"
  local interval="${2:-1}"

  echo "timestamp,gpu_index,util_gpu_pct,mem_used_MiB,mem_total_MiB,power_W,temp_C" > "$out_csv"
  ( while true; do
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
        --format=csv,noheader,nounits \
      | awk -v ts="$(date +%Y-%m-%dT%H:%M:%S)" -F',' '{gsub(/^[ \t]+|[ \t]+$/, "", $0); printf "%s,%s\n", ts, $0}'
      sleep "$interval"
    done ) >> "$out_csv" &
  echo $!
}

gpu_monitor_stop() {
  local pid="$1"
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

gpu_monitor_summarize() {
  local out_csv="$1"
  local summary_txt="$2"

  awk -F',' '
    NR==1 {next}
    {
      util=$3; mem=$4; pwr=$6; temp=$7;
      if(util>max_util) max_util=util;
      if(mem>max_mem)  max_mem=mem;
      sum_util+=util; n++;
      sum_pwr+=pwr;  if(pwr>max_pwr) max_pwr=pwr;
      sum_temp+=temp; if(temp>max_temp) max_temp=temp;
    }
    END{
      if(n==0){print "No GPU samples recorded."; exit 0;}
      printf "GPU samples: %d\n", n;
      printf "Peak GPU memory used (MiB): %.0f\n", max_mem;
      printf "Average GPU utilization (%%): %.2f\n", sum_util/n;
      printf "Peak GPU utilization (%%): %.0f\n", max_util;
      printf "Average power draw (W): %.2f\n", sum_pwr/n;
      printf "Peak power draw (W): %.2f\n", max_pwr;
      printf "Average temperature (C): %.2f\n", sum_temp/n;
      printf "Peak temperature (C): %.0f\n", max_temp;
    }
  ' "$out_csv" > "$summary_txt"
}

run_timed_gpu() {
  local name="$1"; shift
  if [[ "$#" -eq 0 ]]; then
    echo "ERROR: run_timed_gpu(${name}) received no command"
    return 2
  fi
  local cmd_display
  cmd_display="$(printf '%q ' "$@")"
  local ts="$(date +%Y%m%d)"
  local gpu_tag="${CUDA_VISIBLE_DEVICES:-unset}"
  local time_log="${LOG_DIR}/${name}_${ts}_GPU_${gpu_tag}.time.txt"
  local out_log="${LOG_DIR}/${name}_${ts}_GPU_${gpu_tag}.stdout.txt"
  local err_log="${LOG_DIR}/${name}_${ts}_GPU_${gpu_tag}.stderr.txt"
  local gpu_csv="${LOG_DIR}/${name}_${ts}_GPU_${gpu_tag}.gpu.csv"
  local gpu_sum="${LOG_DIR}/${name}_${ts}_GPU_${gpu_tag}.gpu_summary.txt"

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[$(date)] WARNING: nvidia-smi not found; falling back to CPU timing only."
    /usr/bin/time -v -o "$time_log" env -u SHELLOPTS "$@" >"$out_log" 2>"$err_log"
    return
  fi

  echo "[$(date)] TIMED+GPU RUN (${name}): ${cmd_display% }"
  local mon_pid
  mon_pid="$(gpu_monitor_start "$gpu_csv" 1)"

  set +e
  /usr/bin/time -v -o "$time_log" env -u SHELLOPTS "$@" >"$out_log" 2>"$err_log"
  local rc=$?
  set -e

  gpu_monitor_stop "$mon_pid"
  gpu_monitor_summarize "$gpu_csv" "$gpu_sum"

  echo "[$(date)] TIME summary: $time_log"
  echo "[$(date)] GPU summary: $gpu_sum"

  return "$rc"
}

ensure_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: ${label} not found: ${path}"
    exit 2
  fi
}

ensure_dir() {
  local path="$1"
  local label="$2"
  if [[ -z "$path" ]]; then
    echo "ERROR: ${label} path is empty"
    exit 2
  fi
  mkdir -p "$path" || { echo "ERROR: cannot create ${label}: ${path}"; exit 2; }
}

# ========= Running with Timed Log =========
## Set up the directories
# Normalizer and Dictionary
CLR_DIR="${CLR_DIR:-umls_normalizer.cli}"
INPUT_UMLS_REL="${INPUT_UMLS_REL:-umls-normalizer-main/examples/input_genie_step3/}"
OUTPUT_UMLS_REL="${OUTPUT_UMLS_REL:-umls-normalizer-main/examples/output_genie_step3/}"

INPUT_UMLS_DER="${UMLS_DIR}/${INPUT_UMLS_REL#/}"
OUTPUT_UMLS_DER="${UMLS_DIR}/${OUTPUT_UMLS_REL#/}"

# Script location (hardcoded under GENIE_BASE)
SCRIPTS_DIR="${SCRIPTS_DIR:-${GENIE_BASE}/Scripts_Refined_0121}"

# Inputs/outputs relative to GENIE_BASE (portable)
## Step 1: A_jsonl2table (input -> output)
INPUT_JSONL_REL="${INPUT_JSONL_REL:-Data/genie_train.jsonl}"
INPUT_JSONL_ABS="${INPUT_JSONL_ABS:-$GENIE_BASE/$INPUT_JSONL_REL}"
OUTPUT_CSV_REL="${OUTPUT_CSV_REL:-Data/genie_processed.csv}"
OUTPUT_CSV_ABS="${OUTPUT_CSV_ABS:-$GENIE_BASE/$OUTPUT_CSV_REL}"

## Step 2: B_prepare4normalization (input -> output)
NORMAL_CSV_REL="${NORMAL_CSV_REL:-Data/genie_for_normalization.csv}"
NORMAL_CSV_ABS="${NORMAL_CSV_ABS:-$GENIE_BASE/$NORMAL_CSV_REL}"

## Step 3: UMLS normalization + C_normalization (input -> outputs)
UMLS_CSV_REL="${UMLS_CSV_REL:-genie_for_normalization.normalized.csv}"
UMLS_CSV_ABS="${OUTPUT_UMLS_DER%/}/${UMLS_CSV_REL}"
NORMALIZED_CSV_REL="${NORMALIZED_CSV_REL:-Data/genie_normalized.csv}"
NORMALIZED_CSV_ABS="${NORMALIZED_CSV_ABS:-$GENIE_BASE/$NORMALIZED_CSV_REL}"

## Step 4: D_discretization (input -> output)
DISCRET_CSV_REL="${DISCRET_CSV_REL:-Data/genie_discretized.csv}"
DISCRET_CSV_ABS="${DISCRET_CSV_ABS:-$GENIE_BASE/$DISCRET_CSV_REL}"
DISCRET_STEM="${DISCRET_CSV_REL##*/}"
DISCRET_STEM="${DISCRET_STEM%.csv}"

## Step 5: E_load2tensor (build SPPMI, outputs go to Data)
RUN_TAG="${RUN_TAG:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
PIPE_OUT_DIR="${PIPE_OUT_DIR:-${OUT_DIR}/run_${RUN_TAG}}"
PRE6_OUT_DIR="${PRE6_OUT_DIR:-${GENIE_BASE}/Data}"
mkdir -p "$PIPE_OUT_DIR" "$PRE6_OUT_DIR"

# SPPMI + vocab from E (Data)
SPPMI_NPZ_REL="${SPPMI_NPZ_REL:-Data/${DISCRET_STEM}_sppmi.npz}"
SPPMI_NPZ_ABS="${SPPMI_NPZ_ABS:-$GENIE_BASE/$SPPMI_NPZ_REL}"
E_VOCAB_ABS="${E_VOCAB_ABS:-$PRE6_OUT_DIR/${DISCRET_STEM}_cui_vocab.csv}"

## Step 6: F_2Ddecomp (outputs go to Output)
F_OUT_DIR="${F_OUT_DIR:-$PIPE_OUT_DIR/${DISCRET_STEM}_F_2Ddecomp}"
EMBED_CSV_ABS="${EMBED_CSV_ABS:-$F_OUT_DIR/embeddings.csv}"

## Step 7: G_Top_K_Pairs (input from embeddings CSV, output to Output)
G_PAIRS_ABS="${G_PAIRS_ABS:-$PIPE_OUT_DIR/${DISCRET_STEM}_topk_pairs.csv}"

## Step 8: H_Formality_Arrangement (inputs -> output)
if [[ "${UMLS_CSV_REL}" == Data/* ]]; then
  CODEBOOK_CSV_ABS="${CODEBOOK_CSV_ABS:-$GENIE_BASE/$UMLS_CSV_REL}"
else
  CODEBOOK_CSV_ABS="${CODEBOOK_CSV_ABS:-$GENIE_BASE/Data/$UMLS_CSV_REL}"
fi
H_PAIRS_ABS="${H_PAIRS_ABS:-$G_PAIRS_ABS}"
H_OUT_ABS="${H_OUT_ABS:-$PIPE_OUT_DIR/pairs_explained.csv}"

## Step 9: I_GPT_scoring (input -> outputs)
I_OUT_ABS="${I_OUT_ABS:-$PIPE_OUT_DIR/gpt_scored.csv}"
I_AUDIT_ABS="${I_AUDIT_ABS:-$PIPE_OUT_DIR/gpt_scored.audit.jsonl}"
I_BATCHINPUT_ABS="${I_BATCHINPUT_ABS:-$PIPE_OUT_DIR/gpt_batchinput.jsonl}"

## Step 10: J_Result_evaluation (input -> output)
J_OUT_ABS="${J_OUT_ABS:-$PIPE_OUT_DIR/gpt_scored_labeled.csv}"


# Script derived path (defaults to ./scripts)
SCRIPT_DER_A="${SCRIPT_DER_A:-${SCRIPTS_DIR}/A_jsonl2table_0121.py}"
SCRIPT_DER_B="${SCRIPT_DER_B:-${SCRIPTS_DIR}/B_prepare4normalization_0121.py}"
SCRIPT_DER_C="${SCRIPT_DER_C:-${SCRIPTS_DIR}/C_normalization_0121.py}"
SCRIPT_DER_D="${SCRIPT_DER_D:-${SCRIPTS_DIR}/D_discretization_0121.py}"
SCRIPT_DER_E="${SCRIPT_DER_E:-${SCRIPTS_DIR}/E_load2tensor_0121.py}"
SCRIPT_DER_F="${SCRIPT_DER_F:-${SCRIPTS_DIR}/F_2Ddecomp_GPU_0121.py}"
SCRIPT_DER_G="${SCRIPT_DER_G:-${SCRIPTS_DIR}/G_Top_K_Pairs_0121.py}"
SCRIPT_DER_H="${SCRIPT_DER_H:-${SCRIPTS_DIR}/H_Formality_Arrangement_0121.py}"
SCRIPT_DER_I="${SCRIPT_DER_I:-${SCRIPTS_DIR}/I_GPT_scoring_0121.py}"
SCRIPT_DER_J="${SCRIPT_DER_J:-${SCRIPTS_DIR}/J_Result_evaluation_0121.py}"

# ========= GPU capability checks (one-time) =========
GPU_OK="0"
FAISS_GPU_OK="0"
CUPY_OK="0"
TORCH_CUDA_OK="0"

if [[ "${WANT_GPU}" == "1" ]]; then
  GPU_OK="1"
  FAISS_GPU_OK="$(python - <<'PY'
try:
    import faiss
    ok = hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0
    print("1" if ok else "0")
except Exception:
    print("0")
PY
)"
  CUPY_OK="$(python - <<'PY'
try:
    import cupy
    ok = cupy.cuda.runtime.getDeviceCount() > 0
    if ok:
        # Force NVRTC probe so we fail early when libnvrtc.so.12 is not visible.
        from cupy_backends.cuda.libs import nvrtc
        _ = nvrtc.getVersion()
    print("1" if ok else "0")
except Exception:
    print("0")
PY
)"
  TORCH_CUDA_OK="$(python - <<'PY'
try:
    import torch
    ok = bool(torch.cuda.is_available())
    print("1" if ok else "0")
except Exception:
    print("0")
PY
)"
fi

echo "=== GPU capability checks ==="
echo "GPU_OK=${GPU_OK} FAISS_GPU_OK=${FAISS_GPU_OK} CUPY_OK=${CUPY_OK} TORCH_CUDA_OK=${TORCH_CUDA_OK}"
echo


# ========= Step 1: Put jsonl into table =========
ensure_file "$INPUT_JSONL_ABS" "Step 1 input JSONL"
echo "[$(date)] Step 1. Read .jsonl file and transform to .csv table."
echo "  Input : ${GENIE_BASE}/${INPUT_JSONL_REL}"
echo "  Output: ${OUTPUT_CSV_ABS}"
run_timed "Step_1_jsonl_to_table" python "$SCRIPT_DER_A" --base-dir "$GENIE_BASE" \
    --input-jsonl "$INPUT_JSONL_REL" \
    --output-csv "$OUTPUT_CSV_REL"

# ========= Step 2: Prepare for UMLS normalization =========
ensure_file "$OUTPUT_CSV_ABS" "Step 2 input CSV"
echo "[$(date)] Step 2. Prepare the data in .csv formality for UMLS normalization."
echo "  Input : ${OUTPUT_CSV_ABS}"
echo "  Output: ${NORMAL_CSV_ABS}"
run_timed "Step_2_prepare_for_normalization" python "$SCRIPT_DER_B" --base-dir "$GENIE_BASE" \
    --input-csv "$OUTPUT_CSV_REL" --output-csv "$NORMAL_CSV_REL"


# ========= Step 3: Normalization =========
ensure_file "$NORMAL_CSV_ABS" "Step 3 input CSV"
if [[ "${WANT_GPU}" == "1" && "${FAISS_GPU_OK}" != "1" ]]; then
  echo "WARNING: GPU requested but FAISS GPU not available; using CPU for Step 3."
fi

echo "[$(date)] Step 3. UMLS normalization."
echo "  Input : ${NORMAL_CSV_ABS}"
echo "  Output: ${UMLS_CSV_ABS} (UMLS)"
echo "  Output: ${GENIE_BASE}/${NORMALIZED_CSV_REL} (C normalization)"

## Move the files before further processing
mkdir -p "$INPUT_UMLS_DER" "$OUTPUT_UMLS_DER"
cp -f "$NORMAL_CSV_ABS" "$INPUT_UMLS_DER"

## Need to inspect GPU option
if [[ "${WANT_GPU}" == "1" && "${FAISS_GPU_OK}" == "1" ]]; then
  # GPU + faiss-gpu path
  # Need careful modification, probably let Zongxin try this on Brigham
  echo "GPU-based UMLS normalization."
  run_timed_gpu "Step_3_UMLS_normalization" python -m "$CLR_DIR" \
    --mode folder \
    --input-dir "$INPUT_UMLS_DER" \
    --output-dir "$OUTPUT_UMLS_DER" \
    --entity-column entity \
    --dictionary "$UMLS_DICTIONARY_PATH" \
    --device cuda \
    --top-k 5 --threshold 0.35 
else
  # CPU fallback
  echo "CPU-based UMLS normalization."
  run_timed "Step_3_UMLS_normalization" python -m "$CLR_DIR" \
    --mode folder \
    --input-dir "$INPUT_UMLS_DER" \
    --output-dir "$OUTPUT_UMLS_DER" \
    --entity-column entity \
    --dictionary "$UMLS_DICTIONARY_PATH" \
    --device cpu --no-faiss-gpu \
    --dict-batch-size 8192 --query-batch-size 2048
fi

## Move the file back to the original /Data
cp -f "$UMLS_CSV_ABS" "$GENIE_BASE/Data/"

python "$SCRIPT_DER_C" --base-dir "$GENIE_BASE" \
    --input-norm "$GENIE_BASE/Data/$UMLS_CSV_REL" \
    --input-full "$OUTPUT_CSV_REL" \
    --output-csv "$NORMALIZED_CSV_REL" \
    --align positional

# ========= Step 4: Discretization =========
ensure_file "$NORMALIZED_CSV_ABS" "Step 4 input CSV"
echo "[$(date)] Step 4. Data Discretization."
K_THRESHOLD="${K_THRESHOLD:-10}"
echo "  Input : ${GENIE_BASE}/${NORMALIZED_CSV_REL}"
echo "  Output: ${DISCRET_CSV_ABS}"
run_timed "Step_4_discretization" python "$SCRIPT_DER_D" --base-dir "$GENIE_BASE" \
    --input-csv "$NORMALIZED_CSV_REL" --output-csv "$DISCRET_CSV_REL" --K "$K_THRESHOLD"

# ========= Step 5: Load to SPPMI Matrix =========
E_USE_GPU="${E_USE_GPU:-auto}"
E_STATUS_FILTER="${E_STATUS_FILTER:-ALL}"
E_PRESENT_ONLY="${E_PRESENT_ONLY:-0}"
E_TOKEN_DELIM="${E_TOKEN_DELIM:-||}"
E_MIN_CUI_COUNT="${E_MIN_CUI_COUNT:-1}"
E_MIN_COOCCUR="${E_MIN_COOCCUR:-1}"
E_MAX_CUIS_PER_PATIENT="${E_MAX_CUIS_PER_PATIENT:-0}"
E_DTYPE="${E_DTYPE:-float32}"
E_PATIENT_COL="${E_PATIENT_COL:-patient_id}"
E_CUI_COL="${E_CUI_COL:-umls_top_cui}"
E_STATUS_COL="${E_STATUS_COL:-assertion_status_norm}"

if [[ "${E_PRESENT_ONLY}" == "1" ]]; then
  E_STATUS_FILTER="PRESENT"
fi

if [[ "${E_USE_GPU}" == "auto" ]]; then
  if [[ "${WANT_GPU}" == "1" && "${CUPY_OK}" == "1" ]]; then
    E_USE_GPU=1
  else
    E_USE_GPU=0
  fi
fi
if [[ "${E_USE_GPU}" == "1" && "${CUPY_OK}" != "1" ]]; then
  echo "WARNING: E_USE_GPU=1 but CuPy not available; using CPU for Step 5."
  E_USE_GPU=0
fi

if [[ "${E_USE_GPU}" == "1" ]]; then
  E_DEVICE="cuda"
  E_RUNNER="run_timed_gpu"
else
  E_DEVICE="cpu"
  E_RUNNER="run_timed"
fi

ensure_file "$DISCRET_CSV_ABS" "Step 5 input CSV"
ensure_dir "$(dirname "$SPPMI_NPZ_ABS")" "Step 5 output dir"
ensure_dir "$(dirname "$E_VOCAB_ABS")" "Step 5 vocab dir"

echo "[$(date)] Step 5. Build SPPMI from discretized CSV."
echo "  Input : ${DISCRET_CSV_ABS}"
echo "  Output: ${SPPMI_NPZ_ABS}"
echo "  Output: ${E_VOCAB_ABS}"
E_ARGS=(
  --input_csv "$DISCRET_CSV_ABS"
  --output_npz "$SPPMI_NPZ_ABS"
  --vocab_csv "$E_VOCAB_ABS"
  --patient_col "$E_PATIENT_COL"
  --cui_col "$E_CUI_COL"
  --status_col "$E_STATUS_COL"
  --status_filter "$E_STATUS_FILTER"
  --min_cui_count "$E_MIN_CUI_COUNT"
  --min_cooccur "$E_MIN_COOCCUR"
  --max_cuis_per_patient "$E_MAX_CUIS_PER_PATIENT"
  --dtype "$E_DTYPE"
  --device "$E_DEVICE"
  --token_delim "$E_TOKEN_DELIM"
)
if [[ "${E_PRESENT_ONLY}" == "1" ]]; then
  E_ARGS+=(--present_only)
fi

"$E_RUNNER" "Step_5_E_load2tensor" python "$SCRIPT_DER_E" \
  "${E_ARGS[@]}"

# ========= Step 6: F_2Ddecomp (CPU/GPU) =========
DECOMP_USE_GPU="${DECOMP_USE_GPU:-auto}"
DECOMP_K="${DECOMP_K:-128}"
DECOMP_TOL="${DECOMP_TOL:-1e-3}"
DECOMP_MAXITER="${DECOMP_MAXITER:-2000}"
DECOMP_NCV="${DECOMP_NCV:-512}"
DECOMP_WHICH="${DECOMP_WHICH:-LA}"
DECOMP_LOBPCG_NITER="${DECOMP_LOBPCG_NITER:-200}"
DECOMP_LOBPCG_INIT="${DECOMP_LOBPCG_INIT:-randn}"
DECOMP_SEED="${DECOMP_SEED:-0}"
DECOMP_THREADS="${DECOMP_THREADS:-${SLURM_CPUS_PER_TASK:-}}"

if [[ "${DECOMP_USE_GPU}" == "auto" ]]; then
  if [[ "${WANT_GPU}" == "1" && "${TORCH_CUDA_OK}" == "1" ]]; then
    DECOMP_USE_GPU=1
  else
    DECOMP_USE_GPU=0
  fi
fi
if [[ "${DECOMP_USE_GPU}" == "1" && "${TORCH_CUDA_OK}" != "1" ]]; then
  echo "WARNING: DECOMP_USE_GPU=1 but torch.cuda not available; using CPU for Step 6."
  DECOMP_USE_GPU=0
fi

if [[ "${DECOMP_USE_GPU}" == "1" ]]; then
  DECOMP_SOLVER="${DECOMP_SOLVER:-torch_lobpcg}"
  DECOMP_DEVICE="${DECOMP_DEVICE:-cuda}"
  DECOMP_DTYPE="${DECOMP_DTYPE:-float32}"
  DECOMP_RUNNER="run_timed_gpu"
else
  DECOMP_SOLVER="${DECOMP_SOLVER:-eigsh}"
  DECOMP_DEVICE="${DECOMP_DEVICE:-cpu}"
  DECOMP_DTYPE="${DECOMP_DTYPE:-float64}"
  DECOMP_RUNNER="run_timed"
fi

ensure_file "$SPPMI_NPZ_ABS" "Step 6 input SPPMI"
ensure_dir "$F_OUT_DIR" "Step 6 output dir"
if [[ -n "${EMBED_CSV_ABS}" ]]; then
  ensure_file "$E_VOCAB_ABS" "Step 6 vocab CSV"
  ensure_dir "$(dirname "$EMBED_CSV_ABS")" "Step 6 embeddings CSV dir"
fi

echo "[$(date)] Step 6. F_2Ddecomp: sparse decomposition."
echo "  Input : ${SPPMI_NPZ_ABS}"
echo "  Output: ${F_OUT_DIR}"
if [[ -n "${EMBED_CSV_ABS}" ]]; then
  echo "  Output: ${EMBED_CSV_ABS}"
fi
F_ARGS=(
  --input_npz "$SPPMI_NPZ_ABS"
  --out_dir "$F_OUT_DIR"
  --symmetrize
  --drop_diagonal
  --k "$DECOMP_K"
  --tol "$DECOMP_TOL"
  --dtype "$DECOMP_DTYPE"
  --solver "$DECOMP_SOLVER"
  --device "$DECOMP_DEVICE"
  --lobpcg_niter "$DECOMP_LOBPCG_NITER"
  --lobpcg_init "$DECOMP_LOBPCG_INIT"
  --seed "$DECOMP_SEED"
)
if [[ -n "${EMBED_CSV_ABS}" ]]; then
  F_ARGS+=(
    --vocab_csv "$E_VOCAB_ABS"
    --output_csv "$EMBED_CSV_ABS"
    --emb_prefix "emb_"
    --cui_col "token"
    --out_cui_col "pair_token"
  )
fi
if [[ "$DECOMP_SOLVER" == "eigsh" ]]; then
  F_ARGS+=(--maxiter "$DECOMP_MAXITER" --ncv "$DECOMP_NCV" --which "$DECOMP_WHICH")
fi
if [[ -n "${DECOMP_THREADS}" ]]; then
  F_ARGS+=(--threads "$DECOMP_THREADS")
fi
"$DECOMP_RUNNER" "Step_6_F_2Ddecomp" python "$SCRIPT_DER_F" "${F_ARGS[@]}"

# ========= Step 7: G_Top_K_Pairs =========
G_K_SEARCH="${G_K_SEARCH:-2000}"
G_TOP_PP="${G_TOP_PP:-500}"
G_TOP_OTHER="${G_TOP_OTHER:-50}"
G_USE_GPU="${G_USE_GPU:-auto}"

if [[ "${G_USE_GPU}" == "auto" ]]; then
  if [[ "${WANT_GPU}" == "1" && "${FAISS_GPU_OK}" == "1" ]]; then
    G_USE_GPU=1
  else
    G_USE_GPU=0
  fi
fi
if [[ "${G_USE_GPU}" == "1" && "${FAISS_GPU_OK}" != "1" ]]; then
  echo "WARNING: G_USE_GPU=1 but FAISS GPU not available; using CPU for Step 7."
  G_USE_GPU=0
fi

if [[ "${G_USE_GPU}" == "1" ]]; then
  G_RUNNER="run_timed_gpu"
else
  G_RUNNER="run_timed"
fi

ensure_file "$EMBED_CSV_ABS" "Step 7 input embeddings CSV"
ensure_dir "$(dirname "$G_PAIRS_ABS")" "Step 7 output dir"

echo "[$(date)] Step 7. G_Top_K_Pairs: top-K pairs from embeddings."
echo "  Input : ${EMBED_CSV_ABS}"
echo "  Output: ${G_PAIRS_ABS}"
if [[ "${G_USE_GPU}" == "1" ]]; then
  "$G_RUNNER" "Step_7_G_topk_pairs" python "$SCRIPT_DER_G" \
    --input_csv "$EMBED_CSV_ABS" \
    --output_csv "$G_PAIRS_ABS" \
    --k_search "$G_K_SEARCH" --top_pp "$G_TOP_PP" --top_other "$G_TOP_OTHER" \
    --auto_sep --faiss-gpu
else
  "$G_RUNNER" "Step_7_G_topk_pairs" python "$SCRIPT_DER_G" \
    --input_csv "$EMBED_CSV_ABS" \
    --output_csv "$G_PAIRS_ABS" \
    --k_search "$G_K_SEARCH" --top_pp "$G_TOP_PP" --top_other "$G_TOP_OTHER" \
    --auto_sep
fi

# ========= Step 8: H_Formality_Arrangement =========
ensure_file "$H_PAIRS_ABS" "Step 8 input pairs"
ensure_file "$CODEBOOK_CSV_ABS" "Step 8 input codebook"
ensure_dir "$(dirname "$H_OUT_ABS")" "Step 8 output dir"
echo "[$(date)] Step 8. H_Formality_Arrangement: enrich pairs with terms/types."
echo "  Input : ${H_PAIRS_ABS}"
echo "  Input : ${CODEBOOK_CSV_ABS}"
echo "  Output: ${H_OUT_ABS}"
run_timed "Step_8_H_formality" python "$SCRIPT_DER_H" \
  --pairs-csv "$H_PAIRS_ABS" \
  --codebook-csv "$CODEBOOK_CSV_ABS" \
  --out-csv "$H_OUT_ABS"

# ========= Step 9: I_GPT_scoring =========
GPT_MODEL="${GPT_MODEL:-gpt-5.2}"
GPT_MODE="${GPT_MODE:-sync}"
GPT_CONCURRENCY="${GPT_CONCURRENCY:-10}"
GPT_MAX_RETRIES="${GPT_MAX_RETRIES:-5}"
GPT_POLL_SECONDS="${GPT_POLL_SECONDS:-10}"
KEY_FILE="${KEY_FILE:-${GENIE_BASE}/key.txt}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  if [[ -f "$KEY_FILE" ]]; then
    export OPENAI_API_KEY="$(tr -d '\r\n' < "$KEY_FILE")"
  else
    echo "ERROR: OPENAI_API_KEY not set and key file not found: $KEY_FILE"
    exit 2
  fi
fi

ensure_file "$H_OUT_ABS" "Step 9 input pairs"
ensure_dir "$(dirname "$I_OUT_ABS")" "Step 9 output dir"
echo "[$(date)] Step 9. Fetch the score pairs with ChatGPT 5.2."
echo "  Input : ${H_OUT_ABS}"
echo "  Output: ${I_OUT_ABS}"
echo "  Audit : ${I_AUDIT_ABS}"
if [[ "$GPT_MODE" == "batch" ]]; then
  run_timed "Step_9_I_gpt_scoring" python "$SCRIPT_DER_I" \
    --input "$H_OUT_ABS" \
    --output "$I_OUT_ABS" \
    --model "$GPT_MODEL" \
    --mode batch \
    --batchinput "$I_BATCHINPUT_ABS" \
    --poll-seconds "$GPT_POLL_SECONDS" \
    --audit-jsonl "$I_AUDIT_ABS"
else
  run_timed "Step_9_I_gpt_scoring" python "$SCRIPT_DER_I" \
    --input "$H_OUT_ABS" \
    --output "$I_OUT_ABS" \
    --model "$GPT_MODEL" \
    --mode sync \
    --concurrency "$GPT_CONCURRENCY" \
    --max-retries "$GPT_MAX_RETRIES" \
    --audit-jsonl "$I_AUDIT_ABS"
fi

# ========= Step 10: J_Result_evaluation =========
J_MODE="${J_MODE:-tristate}"
J_THRESHOLD="${J_THRESHOLD:-50}"
J_LOW="${J_LOW:-20}"
J_HIGH="${J_HIGH:-80}"

ensure_file "$I_OUT_ABS" "Step 10 input scores"
ensure_dir "$(dirname "$J_OUT_ABS")" "Step 10 output dir"
echo "[$(date)] Step 10. Result Evaluation"
echo "  Input : ${I_OUT_ABS}"
echo "  Output: ${J_OUT_ABS}"
run_timed "Step_10_J_result_evaluation" python "$SCRIPT_DER_J" \
  --input "$I_OUT_ABS" \
  --output "$J_OUT_ABS" \
  --mode "$J_MODE" \
  --threshold "$J_THRESHOLD" \
  --low "$J_LOW" --high "$J_HIGH"
