# WhisperFlow

Local AI-powered voice dictation with auto-editing, context awareness, and command mode. A self-hosted alternative to cloud-based dictation services.

## Features

- **Hold-to-Talk Dictation** — Press and hold `Ctrl+Space`, speak naturally, release. Polished text appears in your active app.
- **AI Auto-Editing** — Removes filler words (um, uh, like), adds punctuation, fixes grammar, handles course corrections ("tomorrow, no wait, Friday" → "Friday").
- **Context-Aware Formatting** — Detects your active app and adjusts tone (professional for email, casual for Slack, technical for code editors).
- **Command Mode** — Hold `Ctrl+Shift+Space` and speak commands: "make this more formal", "translate to Spanish", "summarize this", "turn this into bullet points".
- **100+ Languages** — Powered by OpenAI Whisper large-v3 with support for 99 languages.
- **Personal Dictionary** — Add custom words, names, and jargon for better recognition.
- **Voice Snippets** — Create shortcuts: say "insert my calendar link" to paste saved text.
- **Style Learning** — Learns your writing preferences from corrections over time.
- **AI Assistant** — Ask questions directly: "what is the capital of France" and get answers injected into your text field.
- **Universal App Support** — Works in any text field on Linux (X11) via clipboard injection.
- **GPU Offloading** — Run transcription on a remote GPU server for faster processing.
- **Privacy First** — Everything runs locally. No data leaves your network.

## Architecture

```
Linux VM (Client)                    Windows PC (GPU Server, optional)
┌─────────────────────┐              ┌───────────────────────┐
│ Hotkey → Audio      │   network    │ faster-whisper (CUDA) │
│ → Whisper → LLM     │◄────────────►│ on RTX 5090           │
│ → Text Injection    │              └───────────────────────┘
│ System Tray UI      │
└─────────────────────┘
```

## Quick Start

### 1. Install

```bash
git clone <this-repo> && cd whisperflow
bash scripts/install.sh
```

### 2. Set Up LLM (required for AI features)

**Option A — Ollama (recommended, free, local):**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.1:8b
```

**Option B — Bring Your Own API Key:**
Edit `~/.config/whisperflow/config.yaml`:
```yaml
llm:
  provider: openai  # or anthropic
  openai:
    api_key: "sk-..."
```

### 3. Run

```bash
whisperflow
```

### 4. Use

| Action | Hotkey |
|---|---|
| Dictate | Hold `Ctrl+Space` |
| Command mode | Hold `Ctrl+Shift+Space` |
| Toggle whisper mode | `Ctrl+Alt+W` |

## GPU Server (Optional)

If you have a GPU on another machine (e.g., RTX 5090 on Windows):

**On the Windows PC:**
```
scripts\setup_gpu_server.bat
```

**On the Linux VM, edit** `~/.config/whisperflow/config.yaml`:
```yaml
transcription:
  backend: remote
  remote_url: "http://192.168.1.100:8765"
```

## Configuration

All settings are in `~/.config/whisperflow/config.yaml`. Key options:

| Setting | Description |
|---|---|
| `transcription.model_size` | Whisper model: tiny/base/small/medium/large-v3/large-v3-turbo |
| `transcription.backend` | `local` or `remote` |
| `transcription.language` | `auto` or ISO code (en, es, fr, de, etc.) |
| `llm.provider` | `ollama`, `openai`, `anthropic`, or `none` |
| `processing.auto_edit` | Enable/disable AI text cleanup |
| `processing.tone_matching` | Enable/disable per-app tone adjustment |
| `injection.method` | `xdotool` (X11) or `wl-copy` (Wayland) |

## Command Mode Examples

Select text, then hold `Ctrl+Shift+Space` and say:

- "Make this more formal"
- "Translate to Spanish"
- "Turn this into bullet points"
- "Summarize this in two sentences"
- "Fix the grammar"
- "Make this shorter"
- "Bold the title" (in apps that support it)

## Dependencies

**System:** xdotool, xclip, portaudio
**Python:** faster-whisper, PyQt6, pynput, sounddevice, silero-vad, PyYAML
**LLM:** Ollama (free) or OpenAI/Anthropic API key

## License

MIT
