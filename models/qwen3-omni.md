# Qwen3-Omni 30B-A3B (Alibaba/Qwen)

## Specs

| Field | Value |
|-------|-------|
| Model | Qwen/Qwen3-Omni-30B-A3B-Instruct |
| Total params | ~34.5B |
| Active params | ~4.8B (MoE) |
| Thinker (reasoning) | 30B total, 3B active |
| Talker (speech synthesis) | 3B total, 0.3B active |
| VRAM (BF16) | ~70 GB |
| VRAM (AWQ-8bit) | ~35 GB |
| First packet latency | 234ms (datacenter), ~500-1000ms (Jetson Thor estimated) |
| Voice input languages | 19 |
| Voice output languages | 10 (EN, ES, FR, DE, RU, IT, PT, JA, KO, ZH) |
| Voices | Ethan, Chelsie, Aiden |
| Output sample rate | 24 kHz |
| License | Apache 2.0 |

## What It Replaces

```
BEFORE:  Whisper → Ollama gemma3:4b → Kokoro TTS  (3 models, ~8 GB)
AFTER:   Qwen3-Omni                                (1 model, ~70 GB)
```

## Advantages

- **A single model replaces the full STT + LLM + TTS pipeline**
- Understands tone, emotion, and prosody from audio — not just transcribed text
- MoE: only 4.8B params active per token = efficient for its size
- Better WER than GPT-4o-Transcribe on LibriSpeech (1.7%)
- Competitive reasoning with Qwen3-30B text-only
- 19 input languages including ES
- Loaded in just 26 seconds on Jetson Thor (memory-mapped)
- With AWQ-8bit it would fit comfortably leaving RAM for other services
- Multimodal model: also accepts images and video

## Disadvantages

- **70 GB in BF16** — takes up more than half of Jetson Thor's RAM
- **Only 3 predefined voices** — no voice cloning
- **Audio cannot be streamed yet** — it is generated completely before being returned
- Estimated latency on Jetson Thor: 500ms-1000ms per response (slower than cascaded pipeline)
- WebSocket protocol incompatible with voice-chat — needs its own server
- Batch inference only for audio (no batching for multiple users)
- `disable_talker()` saves ~10 GB if you only want STT+text
- Quantized variants (AWQ-4bit/8bit) are community-made, not official

## Issues Found

1. **Massive download** — ~52 GB including all safetensors + codec + Talker. Took 31 minutes
2. **accelerate warning** — "Some parameters are on the meta device because they were offloaded to cpu" — works but indicates part of the model is on CPU
3. **Custom server needed** — The voice-chat WebSocket protocol is not compatible. A dedicated FastAPI server must be written
4. **NPM config generation bug** — Proxy hosts created via API did not generate nginx configs automatically. They had to be created manually

## Configuration

```python
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    dtype="auto",
    device_map="auto",
    attn_implementation="sdpa",
)
processor = Qwen3OmniMoeProcessor.from_pretrained("Qwen/Qwen3-Omni-30B-A3B-Instruct")

# Voice-to-voice
conversation = [
    {"role": "system", "content": "You are a voice assistant."},
    {"role": "user", "content": [{"type": "audio", "audio": "input.wav"}]},
]
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
inputs = processor(text=text, audio=audios, images=images, videos=videos,
                   return_tensors="pt", padding=True, use_audio_in_video=False)
inputs = inputs.to(model.device).to(model.dtype)

text_ids, audio = model.generate(
    **inputs, speaker="Chelsie", return_audio=True,
    thinker_return_dict_in_generate=True, use_audio_in_video=False,
)
```

## Installation on Jetson Thor

Qwen3-Omni does **NOT** use voice-chat. It has its own standalone FastAPI server on port **8006** because the WebSocket protocol is incompatible.

```bash
# 1. Create project directory
mkdir -p ~/workspace/qwen3-omni-voice/code

# 2. Create venv (can be fresh — Qwen3-Omni doesn't need NeMo or RealtimeSTT)
python3 -m venv ~/workspace/qwen3-omni-voice/venv

# 3. Install dependencies
~/workspace/qwen3-omni-voice/venv/bin/pip install \
  torch --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130

~/workspace/qwen3-omni-voice/venv/bin/pip install \
  transformers accelerate soundfile numpy fastapi uvicorn websockets qwen-omni-utils

# 4. Place server.py in code/
# The server is a standalone FastAPI app with WebSocket at /ws
# (see the Configuration section above for the model loading code)

# 5. Pre-download the model (~52 GB, takes ~31 minutes)
~/workspace/qwen3-omni-voice/venv/bin/python3 -c "
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
Qwen3OmniMoeForConditionalGeneration.from_pretrained('Qwen/Qwen3-Omni-30B-A3B-Instruct', dtype='auto', device_map='auto', attn_implementation='sdpa')
Qwen3OmniMoeProcessor.from_pretrained('Qwen/Qwen3-Omni-30B-A3B-Instruct')
print('Download complete')
"

# 6. Create systemd service
sudo tee /etc/systemd/system/qwen3omni-voice.service << 'EOF'
[Unit]
Description=Qwen3-Omni Voice Chat (port 8006)
After=network.target

[Service]
Type=simple
User=<your-user>
WorkingDirectory=/home/<your-user>/workspace/qwen3-omni-voice/code
Environment=PATH=/home/<your-user>/workspace/qwen3-omni-voice/venv/bin:/usr/local/bin:/usr/bin
ExecStart=/home/<your-user>/workspace/qwen3-omni-voice/venv/bin/python3 server.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now qwen3omni-voice

# 7. Verify
sudo journalctl -u qwen3omni-voice -f
# Wait for "Qwen3-Omni loaded in Xs" then "Uvicorn running on http://0.0.0.0:8006"
# Model loads in ~26 seconds (memory-mapped from cache)
# First load (uncached) takes ~31 minutes for download

# WARNING: Qwen3-Omni uses ~70 GB RAM in BF16. You may need to stop other services:
# sudo systemctl stop whisper-small canary-voice
```

## Verdict

The future of voice assistants. A single model that listens, thinks, and speaks. It fits on Jetson Thor in BF16 but consumes a lot of RAM. Ideal for a dedicated assistant where quality matters more than running multiple services in parallel. With AWQ-8bit it would be more practical for daily use.
