#!/usr/bin/env bash
set -euo pipefail

# setup.sh - convenience installer for Python deps (PyTorch + PyG + requirements)
# Usage:
#   ./setup.sh [cpu|cu118|cu121] [--venv PATH]
# Examples:
#   ./setup.sh cpu
#   ./setup.sh cu118 --venv .venv
# Notes:
# - This script only *creates* a virtualenv if you pass --venv, it does not
#   attempt to manage system packages or CUDA drivers.
# - PyTorch and PyG wheels must match your CUDA and Python versions. This
#   script picks the official PyTorch pip index and the PyG wheel index for
#   the selected platform tag.

PLATFORM=${1:-cpu}
shift || true

VENV_PATH=""
if [[ "${1:-}" == "--venv" ]]; then
  VENV_PATH=${2:-.venv}
fi

if [[ "$PLATFORM" != "cpu" && "$PLATFORM" != cu118 && "$PLATFORM" != cu121 ]]; then
  echo "Unknown platform: $PLATFORM"
  echo "Supported: cpu, cu118, cu121"
  exit 1
fi

if [[ -n "$VENV_PATH" ]]; then
  echo "Creating virtualenv at $VENV_PATH"
  python -m venv "$VENV_PATH"
  # shellcheck disable=SC1091
  source "$VENV_PATH/bin/activate"
fi

echo "Installing PyTorch for platform: $PLATFORM"
if [[ "$PLATFORM" == "cpu" ]]; then
  pip install --upgrade pip
  pip install --no-cache-dir "torch" "torchvision" "torchaudio" --index-url https://download.pytorch.org/whl/cpu
  WHEEL_TAG=cpu
else
  # Example for CUDA 11.8 / 12.1 - adjust if your CUDA differs
  pip install --upgrade pip
  if [[ "$PLATFORM" == "cu118" ]]; then
    PIP_INDEX_URL="https://download.pytorch.org/whl/cu118"
    WHEEL_TAG=cu118
  else
    PIP_INDEX_URL="https://download.pytorch.org/whl/cu121"
    WHEEL_TAG=cu121
  fi
  pip install --no-cache-dir "torch" "torchvision" "torchaudio" --index-url "$PIP_INDEX_URL"
fi

echo "Detecting installed torch version..."
PY_TORCH_VERSION=$(python -c 'import torch, sys; print(torch.__version__.split("+")[0])')
echo "Detected torch version: $PY_TORCH_VERSION"

echo "Installing PyG backend wheels for torch ${PY_TORCH_VERSION} (+${WHEEL_TAG})"
PYG_WHL_INDEX="https://data.pyg.org/whl/torch-${PY_TORCH_VERSION}+${WHEEL_TAG}.html"
echo "Using PyG wheel index: $PYG_WHL_INDEX"

echo "Installing torch-scatter, torch-sparse, torch-cluster, torch-spline-conv"
pip install --no-cache-dir torch-scatter -f "$PYG_WHL_INDEX"
pip install --no-cache-dir torch-sparse -f "$PYG_WHL_INDEX"
pip install --no-cache-dir torch-cluster -f "$PYG_WHL_INDEX"
pip install --no-cache-dir torch-spline-conv -f "$PYG_WHL_INDEX"

echo "Installing torch-geometric"
pip install --no-cache-dir torch-geometric

echo "Installing remaining Python requirements from requirements.txt"
pip install --no-cache-dir -r requirements.txt

echo "Setup complete. If you created a virtualenv, activate it with:"
if [[ -n "$VENV_PATH" ]]; then
  echo "  source $VENV_PATH/bin/activate"
fi

echo "You can now run training: PYTHONPATH=./src python src/run.py"
