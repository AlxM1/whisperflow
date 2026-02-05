@echo off
REM WhisperFlow GPU Server Setup for Windows
REM ==========================================
REM Run this on your Windows PC with the RTX 5090
REM

echo ===================================
echo  WhisperFlow GPU Server Setup
echo ===================================
echo.

REM Check Python
python --version 2>NUL
if errorlevel 1 (
    echo ERROR: Python not found. Install from https://python.org
    pause
    exit /b 1
)

REM Create venv
set VENV_DIR=%USERPROFILE%\.whisperflow-server\venv
echo Creating virtual environment at: %VENV_DIR%
python -m venv "%VENV_DIR%"
call "%VENV_DIR%\Scripts\activate.bat"

REM Install dependencies
echo.
echo Installing dependencies (this may take a while for CUDA support)...
pip install --upgrade pip
pip install faster-whisper fastapi uvicorn python-multipart numpy

REM Install CUDA-enabled PyTorch if not already
pip install torch --index-url https://download.pytorch.org/whl/cu124

echo.
echo [OK] Dependencies installed
echo.

REM Create launcher
set LAUNCHER=%USERPROFILE%\.whisperflow-server\start_server.bat
(
echo @echo off
echo call "%VENV_DIR%\Scripts\activate.bat"
echo echo Starting WhisperFlow GPU Server...
echo echo The server will listen on port 8765
echo echo Press Ctrl+C to stop
echo python -c "from whisperflow.server.gpu_server import create_app; import uvicorn; uvicorn.run(create_app(), host='0.0.0.0', port=8765)"
) > "%LAUNCHER%"

echo ===================================
echo  Setup Complete!
echo ===================================
echo.
echo To start the GPU server:
echo   %LAUNCHER%
echo.
echo Then on your Linux VM, edit ~/.config/whisperflow/config.yaml:
echo   transcription:
echo     backend: remote
echo     remote_url: "http://YOUR_WINDOWS_IP:8765"
echo.
echo Find your Windows IP with: ipconfig
echo.

pause
