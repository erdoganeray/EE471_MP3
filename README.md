# Dream Canvas – EE471 Mini Project 3

A simple chatbot web app for image generation and LLM replies.
Talk to the chatbot with your voice, ask it for an image prompt, paste the prompt and press the paint button.

- **Figma design:** https://www.figma.com/design/NAlfSFWtHzmRjXeXf9YXOp
- **Recordings:** see [`recordings/`](recordings)

| Feature | How | Model |
|---------|-----|-------|
| Prompt → image (paint button) | `POST /api/image` | [stabilityai/sd-turbo](https://huggingface.co/stabilityai/sd-turbo) (diffusers, 512×512, 2 steps) |
| Mic button → text in the message box | `MediaRecorder` in the browser → `POST /api/transcribe` | [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) |
| Send button → chatbot reply in the chat history | `POST /api/chat` | [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) |

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python setup/download_models.py   # about 5 GB, resumes if the connection drops
python app.py                     # http://127.0.0.1:5000
```

FFmpeg must be on the PATH to decode the browser's audio (`winget install Gyan.FFmpeg`).
The models load in the background when the server starts; the subtitle shows `Loading AI models… (n/3 ready)` until they are ready.

Tested on Windows 11, Python 3.12, PyTorch 2.11 + CUDA 12.8, RTX 3050 Ti Laptop GPU (4 GB).

## How it works

```
 Browser (templates/index.html, static/app.js)                Flask (app.py)            models.py
 ─────────────────────────────────────────────                ──────────────            ─────────
 [mic] MediaRecorder ── webm ──▶ POST /api/transcribe  ──────▶ SpeechToText.transcribe  Whisper turbo
        text ◀───────────────── message box
 [send] chat history ── JSON ──▶ POST /api/chat        ──────▶ Chatbot.reply           SmolLM2-360M-Instruct
        reply ◀──────────────── chat bubble (+ "Use as prompt")
 [paint] prompt ─────── JSON ──▶ POST /api/image       ──────▶ ImageGenerator.generate SD-Turbo
        PNG (base64) ◀───────── image output box
```

| File | Role |
|------|------|
| `app.py` | Flask routes; input checks; JSON errors |
| `models.py` | `LazyModel` base class (load once on first use, one request at a time per model) and the three model classes; `AIModels` groups them |
| `templates/index.html` | Layout from the Figma design: title + icon, image box + paint button, prompt box, chat history, mic / message box / send |
| `static/style.css` | Colors, radii and spacing taken from the Figma `Theme` variables |
| `static/app.js` | Button logic, recording, chat history sent with every turn (the server keeps no state) |
| `setup/download_models.py` | Downloads only the weights the app uses (fp16 SD-Turbo instead of the 10+ GB full repo) |

**Fitting three models in 4 GB of VRAM** (measured on an RTX 3050 Ti):

| Model | Device | Speed | VRAM |
|-------|--------|-------|------|
| SD-Turbo | GPU, fp16, `enable_model_cpu_offload()` (only the running part is on the GPU) | ~2 s per 512×512 image | peak 1.7 GB |
| Whisper large-v3-turbo | GPU, fp16 | ~0.5 s for 13 s of audio (CPU: ~11 s) | ~1.7 GB |
| SmolLM2-360M-Instruct | CPU | ~5 s per reply (about the same as on the GPU) | 0 |

Devices can be changed with `IMAGE_DEVICE`, `CHAT_DEVICE` and `ASR_DEVICE` (`cuda` or `cpu`).
