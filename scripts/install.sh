#!/bin/bash
# WhisperFlow - Installation script for Linux
# ============================================
set -e

echo "==================================="
echo "  WhisperFlow Installation"
echo "==================================="
echo ""

# Check Python version
PYTHON=${PYTHON:-python3}
PY_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
PY_MAJOR=$(echo $PY_VERSION | cut -d. -f1)
PY_MINOR=$(echo $PY_VERSION | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]); then
    echo "ERROR: Python 3.11+ required (found $PY_VERSION)"
    exit 1
fi
echo "[OK] Python $PY_VERSION"

# System dependencies
echo ""
echo "Installing system dependencies..."
if command -v apt-get &>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        xdotool xclip xprop \
        portaudio19-dev \
        libxcb-xinerama0 \
        2>/dev/null || true
    echo "[OK] System packages installed"
elif command -v dnf &>/dev/null; then
    sudo dnf install -y \
        xdotool xclip xprop \
        portaudio-devel \
        2>/dev/null || true
    echo "[OK] System packages installed"
elif command -v pacman &>/dev/null; then
    sudo pacman -S --noconfirm \
        xdotool xclip xorg-xprop \
        portaudio \
        2>/dev/null || true
    echo "[OK] System packages installed"
else
    echo "WARNING: Unknown package manager. Please install manually:"
    echo "  xdotool, xclip, xprop, portaudio development headers"
fi

# Create virtual environment
echo ""
VENV_DIR="${WHISPERFLOW_VENV:-$HOME/.local/share/whisperflow/venv}"
echo "Creating virtual environment at: $VENV_DIR"
$PYTHON -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Install WhisperFlow
echo ""
echo "Installing WhisperFlow and dependencies..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

pip install --upgrade pip -q
pip install -e "$PROJECT_DIR" -q

echo "[OK] WhisperFlow installed"

# Create launcher script
LAUNCHER="$HOME/.local/bin/whisperflow"
mkdir -p "$(dirname "$LAUNCHER")"
cat > "$LAUNCHER" << LAUNCHER_EOF
#!/bin/bash
source "$VENV_DIR/bin/activate"
exec python -m whisperflow.main "\$@"
LAUNCHER_EOF
chmod +x "$LAUNCHER"
echo "[OK] Launcher created at: $LAUNCHER"

# Create config directory
CONFIG_DIR="$HOME/.config/whisperflow"
mkdir -p "$CONFIG_DIR"
if [ ! -f "$CONFIG_DIR/config.yaml" ]; then
    cp "$PROJECT_DIR/config/default_config.yaml" "$CONFIG_DIR/config.yaml"
    echo "[OK] Default config created at: $CONFIG_DIR/config.yaml"
fi

# Create desktop entry for autostart (optional)
DESKTOP_DIR="$HOME/.config/autostart"
mkdir -p "$DESKTOP_DIR"
cat > "$DESKTOP_DIR/whisperflow.desktop" << DESKTOP_EOF
[Desktop Entry]
Type=Application
Name=WhisperFlow
Comment=AI Voice Dictation
Exec=$LAUNCHER
Icon=audio-input-microphone
Terminal=false
Categories=Utility;Accessibility;
X-GNOME-Autostart-enabled=false
DESKTOP_EOF
echo "[OK] Desktop entry created (autostart disabled by default)"

echo ""
echo "==================================="
echo "  Installation Complete!"
echo "==================================="
echo ""
echo "Next steps:"
echo ""
echo "1. SETUP LLM (required for AI features):"
echo "   Option A - Ollama (recommended, free, local):"
echo "     curl -fsSL https://ollama.ai/install.sh | sh"
echo "     ollama pull llama3.1:8b"
echo ""
echo "   Option B - Your own API key:"
echo "     Edit ~/.config/whisperflow/config.yaml"
echo "     Set llm.provider to 'openai' or 'anthropic'"
echo "     Add your API key"
echo ""
echo "2. OPTIONAL - GPU Server (for RTX 5090 on Windows):"
echo "   See scripts/setup_gpu_server.bat on the Windows machine"
echo ""
echo "3. RUN: whisperflow"
echo "   Or: $LAUNCHER"
echo ""
echo "4. USE: Hold Ctrl+Space to dictate"
echo "        Hold Ctrl+Shift+Space for command mode"
echo ""
