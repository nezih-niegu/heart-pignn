#!/usr/bin/env bash
#
# heart-pignn -- single script for the whole project.
#
#   ./run.sh setup       organize files and install the environment with uv
#   ./run.sh check       run the tests and a full pass on synthetic data
#   ./run.sh baseline    how informative main.py's rule tree is on its own
#   ./run.sh train [N]   train for N epochs (default 30), then test and visualize
#   ./run.sh evaluate    evaluate the best checkpoint against the previous run
#   ./run.sh visualize [CLASS]   real-time ECG monitor; optional CLASS (N|S|V|F|Q) seeks a stretch rich in it
#   ./run.sh explain     which conduction nodes dominate attention per class
#   ./run.sh ablations   the four ablation runs, each in its own folder
#   ./run.sh gpu         report whether torch can see a CUDA GPU
#   ./run.sh notebook    open Jupyter with the project environment
#   ./run.sh clean       delete checkpoints, figures and synthetic data
#
# With no argument, runs 'setup' then 'check'.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DATA_DIR="mit-bih-arrhythmia-database-1.0.0"
DEMO_DIR="demo-mitdb"
CKPT_DIR="checkpoints_pignn"
FIG_DIR="figures"

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
warn() { printf '\033[33m!! %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31m!! %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- file layout

organize() {
  say "Organizing files"
  mkdir -p src/heart_pignn tests notebooks

  # Files may arrive flattened into a single folder. This is idempotent: if they
  # are already in place, nothing moves.
  shopt -s nullglob
  for f in *.py; do
    case "$f" in
      test_*.py) mv -f "$f" tests/ ;;
      *)         mv -f "$f" src/heart_pignn/ ;;
    esac
  done
  for f in *.ipynb; do mv -f "$f" notebooks/; done
  shopt -u nullglob

  [ -f pyproject.toml ] || die "pyproject.toml missing from $ROOT"
  [ -f src/heart_pignn/__init__.py ] || die "src/heart_pignn/__init__.py missing"
  [ -f src/heart_pignn/cli.py ] || die "src/heart_pignn/cli.py missing"
  echo "  src/heart_pignn: $(ls src/heart_pignn/*.py | wc -l) modules"
  echo "  tests:           $(ls tests/*.py 2>/dev/null | wc -l) files"
  echo "  notebooks:       $(ls notebooks/*.ipynb 2>/dev/null | wc -l) files"
}

# ------------------------------------------------------------------------- uv

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    echo "found uv $(uv --version | awk '{print $2}')"
    return
  fi
  say "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || die "uv installed but is not on PATH. Open a new terminal and retry."
}

setup() {
  organize
  ensure_uv
  say "Installing dependencies (uv sync)"
  uv sync
  gpu
  say "Ready"
  uv run heart-pignn --help
}

gpu() {
  say "GPU check"
  uv run python - <<'PY'
import torch
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"CUDA available: {p.name}, {p.total_memory/1024**3:.1f} GB, torch {torch.__version__}")
    print("Training will use the GPU automatically, with mixed precision.")
else:
    print(f"No CUDA GPU visible to torch {torch.__version__}. Training will run on CPU.")
    print("If this machine has an NVIDIA GPU, install a CUDA build:")
    print("  uv pip install torch --index-url https://download.pytorch.org/whl/cu124")
PY
}

# --------------------------------------------------------------------- checks

check() {
  say "Tests"
  uv run pytest -q

  say "End-to-end pass on synthetic data"
  # Synthetic records verify that everything runs. They are NOT for metrics:
  # the morphologies are toy Gaussians.
  uv run heart-pignn demo-data --out-dir "$DEMO_DIR" --n-records 12 --n-beats 300
  uv run heart-pignn train \
    --data-root "$DEMO_DIR" --output-dir ckpt-demo \
    --epochs 3 --batch-size 64 --samples-per-epoch 768 \
    --hidden-dim 32 --graph-steps 16 --no-visualize
  warn "The numbers above come from synthetic data. Do not report them."
}

# ----------------------------------------------------------------- real data

require_data() {
  [ -d "$DATA_DIR" ] || die "Cannot find $DATA_DIR/
  Download MIT-BIH from PhysioNet into $ROOT:
      wget -r -N -c -np https://physionet.org/files/mitdb/1.0.0/
  Then rename the folder to $DATA_DIR (it must contain 100.dat, 100.hea, 100.atr, ...)"
  local n
  n=$(ls "$DATA_DIR"/*.dat 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] || die "$DATA_DIR exists but holds no .dat files"
  echo "$DATA_DIR: $n records"
}

require_ckpt() {
  [ -f "$CKPT_DIR/best_model.pt" ] || die "No checkpoint in $CKPT_DIR. Run './run.sh train' first."
}

baseline() {
  require_data
  # Run this BEFORE training: it is the reference point for claiming the network
  # added anything on top of the rules.
  say "Baseline: main.py's tree with no network"
  uv run heart-pignn rule-baseline --data-root "$DATA_DIR"
}

train() {
  require_data
  local epochs="${1:-30}"
  shift 2>/dev/null || true          # drop the epoch count; forward the rest
  say "Training for $epochs epochs"
  echo "Resumes only if the target checkpoint dir already exists; delete it to start fresh."
  echo "Extra flags forwarded: $*"
  uv run heart-pignn train \
    --data-root "$DATA_DIR" --output-dir "$CKPT_DIR" --epochs "$epochs" "$@"
}

evaluate() {
  require_data; require_ckpt
  say "Test evaluation"
  uv run heart-pignn evaluate --data-root "$DATA_DIR" --output-dir "$CKPT_DIR"
}

visualize() {
  local focus="${1:-}"
  local focus_flag=""
  [ -n "$focus" ] && focus_flag="--focus-class $focus"
  require_data; require_ckpt
  say "Rendering the validation ECG monitor"
  uv run heart-pignn visualize $focus_flag \
    --data-root "$DATA_DIR" --output-dir "$CKPT_DIR" \
    --out-dir "$FIG_DIR" --n-beats "${2:-25}" --split val
  echo "Written to $FIG_DIR/"
}

explain() {
  require_data; require_ckpt
  say "Attention per conduction node"
  uv run heart-pignn explain --data-root "$DATA_DIR" --output-dir "$CKPT_DIR"
}

ablations() {
  require_data
  local epochs="${1:-30}"
  # Each ablation needs its own folder, or it would resume from the wrong
  # checkpoint and the result would mean nothing.
  say "Ablation 1/4: no link to main.py (--rule-w 0)"
  uv run heart-pignn train --data-root "$DATA_DIR" --output-dir ckpt-no-rules \
    --epochs "$epochs" --rule-w 0 --no-visualize

  say "Ablation 2/4: no regression head (--recon-w 0)"
  uv run heart-pignn train --data-root "$DATA_DIR" --output-dir ckpt-no-regression \
    --epochs "$epochs" --recon-w 0 --no-visualize

  say "Ablation 3/4: no augmentation"
  uv run heart-pignn train --data-root "$DATA_DIR" --output-dir ckpt-no-augment \
    --epochs "$epochs" --no-augment --no-visualize

  say "Ablation 4/4: fully balanced sampling (alpha=1.0)"
  uv run heart-pignn train --data-root "$DATA_DIR" --output-dir ckpt-alpha1 \
    --epochs "$epochs" --sampler-alpha 1.0 --no-visualize

  say "Summary"
  echo "Compare macro F1 and PRD across each test_runs.json:"
  for d in "$CKPT_DIR" ckpt-no-rules ckpt-no-regression ckpt-no-augment ckpt-alpha1; do
    [ -f "$d/test_runs.json" ] && echo "  $d/test_runs.json"
  done
}

notebook() {
  say "Jupyter"
  uv run --with jupyter jupyter lab notebooks/Modelo4_PIGNN_Attention.ipynb
}

clean() {
  say "Deleting checkpoints, figures and synthetic data"
  rm -rf "$CKPT_DIR" ckpt-demo ckpt-no-rules ckpt-no-regression ckpt-no-augment \
         ckpt-alpha1 "$DEMO_DIR" "$FIG_DIR"
  echo "MIT-BIH data and the .venv are left untouched."
}

usage() { sed -n '3,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

case "${1:-default}" in
  setup)     setup ;;
  check)     check ;;
  gpu)       gpu ;;
  baseline)  baseline ;;
  train)     shift; train "$@" ;;
  evaluate)  evaluate ;;
  visualize) visualize "${2:-}" "${3:-}" ;;
  explain)   explain ;;
  ablations) ablations "${2:-}" ;;
  notebook)  notebook ;;
  clean)     clean ;;
  default)   setup; check ;;
  *)         usage; exit 1 ;;
esac
