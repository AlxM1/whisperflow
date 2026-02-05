#!/bin/bash
# WhisperFlow GPU Server Setup for Linux (alternative)
# Run this if your GPU machine also runs Linux
set -e

echo "==================================="
echo "  WhisperFlow GPU Server Setup"
echo "==================================="
echo ""

PYTHON=${PYTHON:-python3}
VENV_DIR="${HOME}/.whisperflow-server/venv"

echo "Creating virtual environment..."
$PYTHON -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "Installing dependencies..."
pip install --upgrade pip -q
pip install faster-whisper fastapi uvicorn python-multipart numpy -q

echo ""
echo "[OK] GPU server ready"
echo ""
echo "To start the server:"
echo "  source $VENV_DIR/bin/activate"
echo "  python -m whisperflow.server.gpu_server --host 0.0.0.0 --port 8765"
echo ""
echo "Then configure your client:"
echo "  transcription:"
echo "    backend: remote"
echo "    remote_url: \"http://THIS_MACHINE_IP:8765\""
echo ""
