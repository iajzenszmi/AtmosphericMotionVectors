#!/usr/bin/env bash
# setup_run_amv_demo.sh
# End-to-end setup and execution script for the Atmospheric Motion Vector demo.
#
# Usage:
#   chmod +x setup_run_amv_demo.sh
#   ./setup_run_amv_demo.sh
#
# Pydroid/interactive display:
#   ./setup_run_amv_demo.sh --show
#
# Smaller/faster run:
#   ./setup_run_amv_demo.sh --size 128 --step 16 --window 21 --search-radius 8
#
# The Python model is in amv_model_demo.py.

set -eu

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv_amv}"
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Atmospheric Motion Vector demo setup"
echo "Working directory: $SCRIPT_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: $PYTHON_BIN not found. Install Python 3 first." >&2
    exit 1
fi

if [ ! -f "amv_model_demo.py" ]; then
    echo "ERROR: amv_model_demo.py not found in $SCRIPT_DIR" >&2
    exit 1
fi

# Try a virtual environment first. If venv is unavailable, fall back to user install.
USE_VENV=0
if "$PYTHON_BIN" -m venv "$VENV_DIR" >/dev/null 2>&1; then
    USE_VENV=1
    # shellcheck disable=SC1091
    . "$VENV_DIR/bin/activate"
    PY="$VENV_DIR/bin/python"
else
    echo "WARNING: Could not create a virtual environment."
    echo "Falling back to --user package installation."
    PY="$PYTHON_BIN"
fi

echo "Python executable: $PY"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install numpy matplotlib pandas pillow

echo
echo "Running AMV model demo..."
"$PY" amv_model_demo.py "$@"

echo
echo "Done. Inspect the output_amv directory:"
echo "  ls -lh output_amv"
echo "  cat output_amv/run_summary.txt"
