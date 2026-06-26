# Jarvis A.I 🤖

A voice-controlled AI desktop assistant built with Python. Jarvis listens to voice commands, processes them with a local LLM, and responds with text-to-speech — like a personal assistant running entirely on your machine.

---

## Demo

> "Open YouTube" → opens browser  
> "What time is it?" → speaks the current time  
> "Search Wikipedia about Python" → fetches and reads a summary  
> Any other query → answered by the AI model

---

## Features

- 🎙️ Voice input via microphone (Google Speech Recognition)
- 🔊 Text-to-speech output (Google TTS / gTTS)
- 🌐 Open websites by voice (YouTube, Google, Wikipedia, Spotify)
- 📁 Open Windows folders by voice (Documents, Downloads, Desktop, etc.)
- 🕐 Tell current time
- 📖 Wikipedia search and summary
- 📝 Create and delete files by voice
- 🧠 AI responses for general queries using a local LLM
- 🔄 Chat reset command

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Voice Input | `SpeechRecognition` + Google Speech API |
| Text-to-Speech | `gTTS` (Google Text-to-Speech) + `playsound` |
| Primary AI Model | `meta-llama/Llama-2-7b-chat-hf` (local, via HuggingFace Transformers) |
| Fallback AI Model | `microsoft/DialoGPT-medium` (local, via HuggingFace Transformers) |
| ML Framework | `PyTorch` + `HuggingFace Transformers` |
| Wikipedia | `wikipedia` Python package |
| Browser Control | `webbrowser` (stdlib) |
| File System | `os`, `platform` (stdlib) |

---

## AI Models Used

### Primary — Llama 2 7B Chat
- **Model:** `meta-llama/Llama-2-7b-chat-hf`
- **Source:** Meta AI via HuggingFace
- **Type:** Local inference (runs on your machine, no API needed)
- **Size:** ~14GB (float16 on GPU) / ~28GB (float32 on CPU)
- **GPU:** Auto-detected — uses CUDA if available, falls back to CPU
- **Note:** Requires HuggingFace account and Meta approval to download

### Fallback — DialoGPT Medium
- **Model:** `microsoft/DialoGPT-medium`
- **Source:** Microsoft via HuggingFace
- **Type:** Local inference
- **Size:** ~1.5GB
- **Used when:** Llama 2 fails to load (insufficient RAM/GPU)

---

## Project Structure

```
jarvis/
├── main.py          # Main assistant logic
└── requirements.txt # Python dependencies
```

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/jarvis-ai.git
cd jarvis-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install PyAudio (for microphone)

**Windows:**
```bash
pip install pyaudio
```

**Linux:**
```bash
sudo apt-get install portaudio19-dev
pip install pyaudio
```

**Mac:**
```bash
brew install portaudio
pip install pyaudio
```

### 4. HuggingFace login (for Llama 2)

```bash
pip install huggingface_hub
huggingface-cli login
```

> Llama 2 requires Meta approval. Request access at:  
> https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

---

## Requirements

```
torch
transformers
SpeechRecognition
gtts
playsound
wikipedia
huggingface_hub
```

**Hardware requirements:**
- Llama 2 7B: minimum 16GB RAM (GPU recommended, 8GB+ VRAM)
- DialoGPT fallback: minimum 4GB RAM

---

## Usage

```bash
python main.py
```

Jarvis will start and say **"Jarvis A.I"** — it's ready to listen.

### Voice Commands

| Command | Action |
|---|---|
| `open youtube` | Opens YouTube in browser |
| `open google` | Opens Google in browser |
| `open spotify` | Opens Spotify in browser |
| `open wikipedia site` | Opens Wikipedia in browser |
| `open documents` | Opens Documents folder |
| `open downloads` | Opens Downloads folder |
| `open desktop` | Opens Desktop folder |
| `what time is it` | Speaks current time |
| `search wikipedia about [topic]` | Fetches Wikipedia summary |
| `create file [filename]` | Creates an empty file |
| `delete file [filename]` | Deletes a file |
| `reset chat` | Resets AI conversation |
| `jarvis quit` | Shuts down Jarvis |
| Anything else | Answered by the AI model |

---

## How It Works

```
Microphone → Google Speech Recognition → Command Parser
                                               ↓
                              Known command? → Execute directly
                                               ↓
                              Unknown query? → Llama 2 / DialoGPT
                                               ↓
                                          gTTS → Speaker
```

1. `takeCommand()` captures audio and transcribes via Google Speech API
2. The query is matched against known commands (sites, folders, time, Wikipedia)
3. Unrecognized queries are passed to `get_ai_response()` which runs local inference
4. The response is spoken aloud via `say()` using gTTS

---

## Known Limitations

- Llama 2 is slow on CPU — GPU strongly recommended
- gTTS requires an internet connection (uses Google servers)
- Google Speech Recognition requires internet
- Microphone only works locally — not deployable to a server as-is
- Llama 2 download is ~14GB on first run

---

## Future Improvements

- [ ] Replace local Llama with HuggingFace Inference API (lighter, faster)
- [ ] Add React frontend with chat UI
- [ ] Move mic input to browser using Web Speech API
- [ ] Deploy backend to Render, frontend to Vercel
- [ ] Add conversation memory / chat history

---

## Built As

College project — JSS AKTU  
Demonstrating local LLM integration, voice I/O, and Python automation.

---

## License

MIT
