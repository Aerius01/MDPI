#!/usr/bin/env bash
set -euo pipefail

# Resolve project root to the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Locate conda executable (no activation required)
CONDA_BIN="${CONDA_BIN:-}"
if [ -z "$CONDA_BIN" ]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
  elif [ -x "$HOME/miniconda3/bin/conda" ]; then
    CONDA_BIN="$HOME/miniconda3/bin/conda"
  elif [ -x "$HOME/anaconda3/bin/conda" ]; then
    CONDA_BIN="$HOME/anaconda3/bin/conda"
  elif [ -x "$HOME/miniforge3/bin/conda" ]; then
    CONDA_BIN="$HOME/miniforge3/bin/conda"
  elif [ -x "$HOME/mambaforge/bin/conda" ]; then
    CONDA_BIN="$HOME/mambaforge/bin/conda"
  elif [ -x "/opt/conda/bin/conda" ]; then
    CONDA_BIN="/opt/conda/bin/conda"
  fi
fi

if [ -z "$CONDA_BIN" ]; then
  echo "Conda executable not found. Set CONDA_BIN or install Miniconda/Anaconda." >&2
  exit 1
fi

# Choose environment by prefix (path) or name
ENV_NAME="${ENV_NAME:-mdpi-env}"
ENV_PREFIX="${ENV_PREFIX:-}"

# Auto-detect a likely prefix if not explicitly provided
if [ -z "$ENV_PREFIX" ]; then
  if [ -d "$HOME/miniconda3/envs/$ENV_NAME" ]; then
    ENV_PREFIX="$HOME/miniconda3/envs/$ENV_NAME"
  elif [ -d "$HOME/anaconda3/envs/$ENV_NAME" ]; then
    ENV_PREFIX="$HOME/anaconda3/envs/$ENV_NAME"
  elif [ -d "$HOME/miniforge3/envs/$ENV_NAME" ]; then
    ENV_PREFIX="$HOME/miniforge3/envs/$ENV_NAME"
  elif [ -d "$HOME/mambaforge/envs/$ENV_NAME" ]; then
    ENV_PREFIX="$HOME/mambaforge/envs/$ENV_NAME"
  elif [ -d "/opt/conda/envs/$ENV_NAME" ]; then
    ENV_PREFIX="/opt/conda/envs/$ENV_NAME"
  fi
fi

# Build target args for conda run
RUN_TARGET_ARGS=()
if [ -n "$ENV_PREFIX" ]; then
  if [ ! -d "$ENV_PREFIX" ]; then
    echo "ENV_PREFIX set but directory not found: $ENV_PREFIX" >&2
    exit 1
  fi
  RUN_TARGET_ARGS=(-p "$ENV_PREFIX")
else
  RUN_TARGET_ARGS=(-n "$ENV_NAME")
fi

# Optional: allow overriding port with PORT env var
PORT="${PORT:-8501}"

# Quick check that streamlit is installed in the target env
if ! "$CONDA_BIN" run "${RUN_TARGET_ARGS[@]}" python -c "import streamlit" >/dev/null 2>&1; then
  echo "Streamlit not found in the target environment. Install it first." >&2
  echo "Example: \"$CONDA_BIN\" run ${RUN_TARGET_ARGS[*]} pip install streamlit" >&2
  exit 1
fi

echo "Launching Streamlit app from $SCRIPT_DIR/app/app.py on port $PORT..."
echo "Using conda: $CONDA_BIN"
if [ -n "$ENV_PREFIX" ]; then
  echo "Env prefix: $ENV_PREFIX"
else
  echo "Env name: $ENV_NAME"
fi
echo "Press Ctrl+C to stop."

exec "$CONDA_BIN" run --no-capture-output "${RUN_TARGET_ARGS[@]}" \
  streamlit run app/app.py --server.port "$PORT" --server.headless true "$@"


