#!/usr/bin/env bash
set -euo pipefail

SERVICE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SERVICE_DIR/.." && pwd)"
PYTHON="${FLUX2KLEIN_PYTHON:-$REPO_ROOT/.venv/bin/python}"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment not found at $PYTHON; create .venv and install service/requirements.txt." >&2
  exit 1
fi

ACTION="${1:-serve}"
if [[ $# -gt 0 ]]; then shift; fi
export FLUX2KLEIN_API_KEY="${FLUX2KLEIN_API_KEY:-}"
export FLUX2_MODEL_PATH="${FLUX2_MODEL_PATH:-$HOME/models/FLUX.2-klein-9B}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/fastflux2klein/inductor}"

case "$ACTION" in
  serve|start)
    : "${FLUX2KLEIN_API_KEY:?Set FLUX2KLEIN_API_KEY in .env or the environment}"
    exec "$PYTHON" -I -u "$SERVICE_DIR/server.py" \
      --model "$FLUX2_MODEL_PATH" \
      --host "${FLUX2KLEIN_HOST:-127.0.0.1}" \
      --port "${FLUX2KLEIN_PORT:-8036}" \
      --gpu-memory-gb "${FLUX2KLEIN_GPU_MEMORY_GB:-40}" "$@"
    ;;
  generate|edit)
    : "${FLUX2KLEIN_API_KEY:?Set FLUX2KLEIN_API_KEY in .env or the environment}"
    if [[ "$ACTION" == edit ]]; then
      exec "$PYTHON" -I "$SERVICE_DIR/client_flux2klein.py" --edit "$@"
    fi
    exec "$PYTHON" -I "$SERVICE_DIR/client_flux2klein.py" "$@"
    ;;
  benchmark)
    : "${FLUX2KLEIN_API_KEY:?Set FLUX2KLEIN_API_KEY in .env or the environment}"
    exec "$PYTHON" -I "$SERVICE_DIR/benchmark_v2.py" \
      --base-url "http://127.0.0.1:${FLUX2KLEIN_PORT:-8036}" "$@"
    ;;
  status)
    exec curl --fail --silent --show-error \
      "http://127.0.0.1:${FLUX2KLEIN_PORT:-8036}/health"
    ;;
  test)
    exec "$PYTHON" -I "$SERVICE_DIR/test_contract.py"
    ;;
  verify)
    : "${FLUX2KLEIN_API_KEY:?Set FLUX2KLEIN_API_KEY in .env or the environment}"
    export FLUX2KLEIN_BASE_URL="${FLUX2KLEIN_BASE_URL:-http://127.0.0.1:${FLUX2KLEIN_PORT:-8036}}"
    exec "$PYTHON" -I "$SERVICE_DIR/verify_live.py"
    ;;
  help|-h|--help)
    echo 'Usage: service/run.sh [serve|generate|edit|benchmark|status|test|verify] [options]'
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac
