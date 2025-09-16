#!/usr/bin/env bash
set -euo pipefail

# Resolve project root to the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Try to make conda available in this non-interactive shell
if ! command -v conda >/dev/null 2>&1; then
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/anaconda3/etc/profile.d/conda.sh"
  elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    . "/opt/conda/etc/profile.d/conda.sh"
  else
    echo "Conda initialization script not found. Please install Miniconda/Anaconda or adjust this script to your conda path." >&2
    exit 1
  fi
fi

# Verify conda is now available
if ! command -v conda >/dev/null 2>&1; then
  echo "Conda command not available after sourcing. Aborting." >&2
  exit 1
fi

ENV_NAME="mdpi-env"

# Activate the environment
conda activate "$ENV_NAME" || {
  echo "Conda environment '$ENV_NAME' not found. Create it and install dependencies first." >&2
  exit 1
}

# Ensure streamlit is available
if ! command -v streamlit >/dev/null 2>&1; then
  echo "'streamlit' not found in environment '$ENV_NAME'. Install it (e.g., 'pip install streamlit')." >&2
  exit 1
fi

# Optional: allow overriding port with PORT env var
PORT="${PORT:-8501}"

echo "Launching Streamlit app from $SCRIPT_DIR/app/app.py on port $PORT using env '$ENV_NAME'..."
echo "Press Ctrl+C to stop."

exec streamlit run app/app.py --server.port "$PORT" --server.headless true


