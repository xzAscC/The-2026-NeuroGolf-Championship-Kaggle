#!/usr/bin/env bash
set -euo pipefail

# ── NeuroGolf 2026 - Local Setup Script ──────────────────────────────
# Creates a local venv, installs deps, downloads competition data.
# Does NOT touch your global Python environment.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
DATA_DIR="$SCRIPT_DIR/data"

echo "=== NeuroGolf 2026 Setup ==="
echo "Project: $SCRIPT_DIR"
echo ""

# 1. Create virtual environment
if [ -d "$VENV_DIR" ]; then
    echo "[1/4] venv already exists at $VENV_DIR (skipping)"
else
    echo "[1/4] Creating venv at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
    echo "      Done."
fi

# 2. Upgrade pip inside venv
echo "[2/4] Upgrading pip ..."
"$VENV_DIR/bin/pip" install --upgrade pip --quiet

# 3. Install dependencies
echo "[3/4] Installing dependencies (numpy, onnx, onnxruntime, onnx-tool) ..."
"$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/requirements.txt" --quiet
echo "      Done."

# 4. Download competition data
mkdir -p "$DATA_DIR"
DATA_FILE="$DATA_DIR/all_tasks.json"

if [ -f "$DATA_FILE" ]; then
    echo "[4/4] Data file already exists at $DATA_FILE (skipping)"
else
    echo "[4/4] Downloading ARC-AGI task data (~400 tasks) ..."
    curl -L -o "$DATA_FILE" \
        "https://huggingface.co/LuciferMrng/neurogolf-2026/resolve/main/all_tasks.json"
    echo "      Done. ($(du -h "$DATA_FILE" | cut -f1))"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate the environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To run the solver:"
echo "  python solver.py --data_file data/all_tasks.json --output_dir submission --conv_budget 30"
echo ""
echo "To run just a few tasks for testing:"
echo "  python solver.py --data_file data/all_tasks.json --output_dir submission --tasks 0,1,2,3,4"
