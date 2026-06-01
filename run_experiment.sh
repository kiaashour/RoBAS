#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  cat <<'EOF'
Usage:
  ./run_experiment.sh --list
  ./run_experiment.sh <target> [extra python args...]

Examples:
  ./run_experiment.sh synthetic
  ./run_experiment.sh uci/cb --num_trials 20 --n_cal 10
  ./run_experiment.sh image_dataset/cb --dataset_name utkfaces --embeddings_path /path/to/embeddings_dir
EOF
}

list_targets() {
  cat <<'EOF'
Available targets:
  synthetic
  uci_parallel/cb
  uci_parallel/cbma
  uci_parallel/cqr
  uci_parallel/local
  uci_parallel/main_methods
  uci_parallel/cb
  uci_parallel/cbma
  uci_parallel/cqr
  uci_parallel/local
  uci_parallel/main_methods  
  image_dataset/cb
  image_dataset/cbma
  image_dataset/cqr
  image_dataset/local
  image_dataset/main_methods
  image_datasets_parallel/cb
  image_datasets_parallel/cbma
  image_datasets_parallel/cqr
  image_datasets_parallel/local
  image_datasets_parallel/main_methods  
EOF
}

normalize_target() {
  local raw="$1"
  case "$raw" in
    image_datasets/*) echo "image_dataset/${raw#image_datasets/}" ;;
    iamge_dataset/*) echo "image_dataset/${raw#iamge_dataset/}" ;;
    *) echo "$raw" ;;
  esac
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--list" ]]; then
  list_targets
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage
  echo
  list_targets
  exit 1
fi

TARGET="$(normalize_target "$1")"
shift

SCRIPT_PATH=""
DEFAULT_ARGS=()

case "$TARGET" in
  synthetic)
    SCRIPT_PATH="$ROOT_DIR/scripts/synthetic.py"
    DEFAULT_ARGS=()
    ;;

  uci/cb)
    SCRIPT_PATH="$ROOT_DIR/scripts/uci/cb.py"
    DEFAULT_ARGS=(
      --dataset_name airfoil
      --model_results_dir "$ROOT_DIR/results/uci/model_posteriors"
    )
    ;;
  uci/cbma)
    SCRIPT_PATH="$ROOT_DIR/scripts/uci/cbma.py"
    DEFAULT_ARGS=(
      --dataset_name airfoil
      --model_results_dir "$ROOT_DIR/results/uci/model_posteriors"
    )
    ;;
  uci/cqr)
    SCRIPT_PATH="$ROOT_DIR/scripts/uci/cqr.py"
    DEFAULT_ARGS=(
      --dataset_name airfoil
      --model_name rf
    )
    ;;
  uci/local)
    SCRIPT_PATH="$ROOT_DIR/scripts/uci/local.py"
    DEFAULT_ARGS=(
      --dataset_name airfoil
      --model_name rf
    )
    ;;
  uci/main_methods)
    SCRIPT_PATH="$ROOT_DIR/scripts/uci/main_methods.py"
    DEFAULT_ARGS=(
      --dataset_name airfoil
      --model_name rf
    )
    ;;

  uci_parallel/cb)
    SCRIPT_PATH="$ROOT_DIR/scripts/uci_parallel/cb.py"
    DEFAULT_ARGS=(
      --dataset_name airfoil
      --model_results_dir "$ROOT_DIR/results/uci/model_posteriors"
    )
    ;;
  uci_parallel/cbma)
    SCRIPT_PATH="$ROOT_DIR/scripts/uci_parallel/cbma.py"
    DEFAULT_ARGS=(
      --dataset_name airfoil
      --model_results_dir "$ROOT_DIR/results/uci/model_posteriors"
    )
    ;;
  uci_parallel/cqr)
    SCRIPT_PATH="$ROOT_DIR/scripts/uci_parallel/cqr.py"
    DEFAULT_ARGS=(
      --dataset_name airfoil
      --model_name rf
    )
    ;;
  uci_parallel/local)
    SCRIPT_PATH="$ROOT_DIR/scripts/uci_parallel/local.py"
    DEFAULT_ARGS=(
      --dataset_name airfoil
      --model_name rf
    )
    ;;
  uci_parallel/main_methods)
    SCRIPT_PATH="$ROOT_DIR/scripts/uci_parallel/main_methods.py"
    DEFAULT_ARGS=(
      --dataset_name airfoil
      --model_name rf
    )
    ;;    

  image_dataset/cb)
    SCRIPT_PATH="$ROOT_DIR/scripts/image_datasets/cb.py"
    DEFAULT_ARGS=(
      --dataset_name vvolume
    )
    ;;
  image_dataset/cbma)
    SCRIPT_PATH="$ROOT_DIR/scripts/image_datasets/cbma.py"
    DEFAULT_ARGS=(
      --dataset_name vvolume
    )
    ;;
  image_dataset/cqr)
    SCRIPT_PATH="$ROOT_DIR/scripts/image_datasets/cqr.py"
    DEFAULT_ARGS=(
      --dataset_name vvolume
      --model_name rf
    )
    ;;
  image_dataset/local)
    SCRIPT_PATH="$ROOT_DIR/scripts/image_datasets/local.py"
    DEFAULT_ARGS=(
      --dataset_name vvolume
      --model_name rf
    )
    ;;
  image_dataset/main_methods)
    SCRIPT_PATH="$ROOT_DIR/scripts/image_datasets/main_methods.py"
    DEFAULT_ARGS=(
      --dataset_name vvolume
      --model_name rf
    )
    ;;

  image_datasets_parallel/cb)
    SCRIPT_PATH="$ROOT_DIR/scripts/image_datasets_parallel/cb.py"
    DEFAULT_ARGS=(
      --dataset_name vvolume
    )
    ;;
  image_datasets_parallel/cbma)
    SCRIPT_PATH="$ROOT_DIR/scripts/image_datasets_parallel/cbma.py"
    DEFAULT_ARGS=(
      --dataset_name vvolume
    )
    ;;
  image_datasets_parallel/cqr)
    SCRIPT_PATH="$ROOT_DIR/scripts/image_datasets_parallel/cqr.py"
    DEFAULT_ARGS=(
      --dataset_name vvolume
      --model_name rf
    )
    ;;
  image_datasets_parallel/local)
    SCRIPT_PATH="$ROOT_DIR/scripts/image_datasets_parallel/local.py"
    DEFAULT_ARGS=(
      --dataset_name vvolume
      --model_name rf
    )
    ;;
  image_datasets_parallel/main_methods)
    SCRIPT_PATH="$ROOT_DIR/scripts/image_datasets_parallel/main_methods.py"
    DEFAULT_ARGS=(
      --dataset_name vvolume
      --model_name rf
    )
    ;;    

  *)
    echo "Unknown target: $TARGET" >&2
    echo >&2
    list_targets >&2
    exit 1
    ;;
esac

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "Script not found: $SCRIPT_PATH" >&2
  exit 1
fi

echo "Running target: $TARGET"
echo "Python script: $SCRIPT_PATH"
echo "Command: $PYTHON_BIN $SCRIPT_PATH ${DEFAULT_ARGS[*]} $*"
exec "$PYTHON_BIN" "$SCRIPT_PATH" "${DEFAULT_ARGS[@]}" "$@"
