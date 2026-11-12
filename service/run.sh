#!/usr/bin/env bash
set -euo pipefail

SERVICE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SERVICE_DIR/.." && pwd)"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

ACTION="${1:-serve}"
if [[ $# -gt 0 ]]; then shift; fi

usage() {
  cat <<'USAGE'
Usage: service/run.sh <action> [options]

Actions:
  install|setup  Create the Conda environment (if needed) and install dependencies
  serve|start    Start the API in the foreground
  generate       Generate an image with the OpenAI-compatible client
  edit           Edit one or more reference images
  benchmark      Run the HTTP throughput benchmark
  hf             Run the Hugging Face CLI inside the Conda environment
  status         Query the local health endpoint
  test           Run the CPU API contract tests
  verify         Run live GPU integration checks

Conda selection:
  FLUX2KLEIN_CONDA_ENV     Environment name, default: flux2klein
  FLUX2KLEIN_CONDA_PREFIX  Optional absolute environment path; takes precedence
  FLUX2KLEIN_CONDA_EXE     Optional path to the conda executable
USAGE
}

if [[ "$ACTION" == help || "$ACTION" == -h || "$ACTION" == --help ]]; then
  usage
  exit 0
fi

CONDA_BIN="${FLUX2KLEIN_CONDA_EXE:-${CONDA_EXE:-}}"
CONDA_ENV_NAME="${FLUX2KLEIN_CONDA_ENV:-flux2klein}"
CONDA_ENV_PREFIX="${FLUX2KLEIN_CONDA_PREFIX:-}"

resolve_conda() {
  if [[ -n "$CONDA_BIN" && -x "$CONDA_BIN" ]]; then
    return 0
  fi

  if [[ -z "$CONDA_BIN" ]]; then
    CONDA_BIN="$(type -P conda 2>/dev/null || true)"
  fi

  if [[ -z "$CONDA_BIN" ]]; then
    for candidate in \
      "$HOME/miniconda3/bin/conda" \
      "$HOME/anaconda3/bin/conda" \
      "/opt/conda/bin/conda" \
      "/opt/miniconda3/bin/conda"; do
      if [[ -x "$candidate" ]]; then
        CONDA_BIN="$candidate"
        break
      fi
    done
  fi

  if [[ -z "$CONDA_BIN" ]] && command -v conda >/dev/null 2>&1; then
    # A sourced conda shell function is valid even when no executable is on PATH.
    CONDA_BIN="conda"
  fi

  if [[ -z "$CONDA_BIN" ]]; then
    echo "Conda was not found. Load Conda in this shell or set FLUX2KLEIN_CONDA_EXE." >&2
    exit 1
  fi
}

run_in_env() {
  if [[ -n "$CONDA_ENV_PREFIX" ]]; then
    "$CONDA_BIN" run --no-capture-output --prefix "$CONDA_ENV_PREFIX" "$@"
  else
    "$CONDA_BIN" run --no-capture-output --name "$CONDA_ENV_NAME" "$@"
  fi
}

exec_in_env() {
  if [[ -n "$CONDA_ENV_PREFIX" ]]; then
    exec "$CONDA_BIN" run --no-capture-output --prefix "$CONDA_ENV_PREFIX" "$@"
  else
    exec "$CONDA_BIN" run --no-capture-output --name "$CONDA_ENV_NAME" "$@"
  fi
}

environment_exists() {
  if [[ -n "$CONDA_ENV_PREFIX" ]]; then
    [[ -x "$CONDA_ENV_PREFIX/bin/python" ]]
  else
    run_in_env python -c 'import sys; print(sys.executable)' >/dev/null 2>&1
  fi
}

create_environment() {
  if environment_exists; then
    return 0
  fi
  echo "Creating Conda environment ${CONDA_ENV_PREFIX:-$CONDA_ENV_NAME} (Python 3.10)..."
  if [[ -n "$CONDA_ENV_PREFIX" ]]; then
    "$CONDA_BIN" create --prefix "$CONDA_ENV_PREFIX" python=3.10 pip -y
  else
    "$CONDA_BIN" create --name "$CONDA_ENV_NAME" python=3.10 pip -y
  fi
}

require_environment() {
  if ! environment_exists; then
    echo "Conda environment ${CONDA_ENV_PREFIX:-$CONDA_ENV_NAME} is missing." >&2
    echo "Run: $0 install" >&2
    exit 1
  fi
}

install_environment() {
  create_environment
  run_in_env python -m pip install --upgrade pip
  run_in_env python -m pip install -r "$SERVICE_DIR/requirements.txt"
}

export FLUX2KLEIN_API_KEY="${FLUX2KLEIN_API_KEY:-}"
export FLUX2_MODEL_PATH="${FLUX2_MODEL_PATH:-$HOME/models/FLUX.2-klein-9B}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/fastflux2klein/inductor}"
cd "$REPO_ROOT"

case "$ACTION" in
  install|setup|bootstrap)
    resolve_conda
    install_environment
    echo "Conda environment is ready: ${CONDA_ENV_PREFIX:-$CONDA_ENV_NAME}"
    ;;
  serve|start)
    resolve_conda
    require_environment
    : "${FLUX2KLEIN_API_KEY:?Set FLUX2KLEIN_API_KEY in .env or the environment}"
    exec_in_env python -I -u "$SERVICE_DIR/server.py" \
      --model "$FLUX2_MODEL_PATH" \
      --host "${FLUX2KLEIN_HOST:-127.0.0.1}" \
      --port "${FLUX2KLEIN_PORT:-8036}" \
      --gpu-memory-gb "${FLUX2KLEIN_GPU_MEMORY_GB:-40}" "$@"
    ;;
  generate|edit)
    resolve_conda
    require_environment
    : "${FLUX2KLEIN_API_KEY:?Set FLUX2KLEIN_API_KEY in .env or the environment}"
    if [[ "$ACTION" == edit ]]; then
      exec_in_env python -I "$SERVICE_DIR/client_flux2klein.py" --edit "$@"
    fi
    exec_in_env python -I "$SERVICE_DIR/client_flux2klein.py" "$@"
    ;;
  benchmark)
    resolve_conda
    require_environment
    : "${FLUX2KLEIN_API_KEY:?Set FLUX2KLEIN_API_KEY in .env or the environment}"
    exec_in_env python -I "$SERVICE_DIR/benchmark_v2.py" \
      --base-url "http://127.0.0.1:${FLUX2KLEIN_PORT:-8036}" "$@"
    ;;
  hf)
    resolve_conda
    require_environment
    exec_in_env hf "$@"
    ;;
  status)
    exec curl --fail --silent --show-error \
      "http://127.0.0.1:${FLUX2KLEIN_PORT:-8036}/health"
    ;;
  test)
    resolve_conda
    require_environment
    exec_in_env python -I "$SERVICE_DIR/test_contract.py"
    ;;
  verify)
    resolve_conda
    require_environment
    : "${FLUX2KLEIN_API_KEY:?Set FLUX2KLEIN_API_KEY in .env or the environment}"
    export FLUX2KLEIN_BASE_URL="${FLUX2KLEIN_BASE_URL:-http://127.0.0.1:${FLUX2KLEIN_PORT:-8036}}"
    exec_in_env python -I "$SERVICE_DIR/verify_live.py"
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac
