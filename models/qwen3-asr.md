# Qwen3-ASR (Alibaba/Qwen)

## Specs

| Field | 0.6B | 1.7B |
|-------|------|------|
| Model | Qwen/Qwen3-ASR-0.6B | Qwen/Qwen3-ASR-1.7B |
| Architecture | Qwen3 LLM + AuT encoder | Qwen3 LLM + AuT encoder |
| Runtime | qwen-asr (transformers) | qwen-asr (transformers) |
| Device | GPU | GPU |
| VRAM | ~2 GB | ~4 GB |
| Languages | 30 + 22 Chinese dialects | 30 + 22 Chinese dialects |
| License | Apache 2.0 | Apache 2.0 |
| WER LibriSpeech Clean | 2.11 | 1.63 |

## Advantages

- **SOTA open-source in accuracy** — surpasses Whisper large-v3 on all benchmarks
- 30 languages + auto-detection with 97.9% accuracy
- **Real streaming** via vLLM backend (chunked, revisable)
- Accepts audio as a tuple `(np.ndarray, sample_rate)` directly
- Word-level timestamps with a separate ForcedAligner
- Handles code-switching (EN/ES mixing in the same sentence)
- Recognizes singing and audio with background music
- 1.7B loaded in 126s on Thor

## Disadvantages

- **The `qwen-asr` package pins transformers==4.57.6** — may conflict with other models
- Requires `nagisa` and `soynlp` (Japanese/Korean tokenization deps) even if you don't use those languages
- No ONNX export — only runs via transformers or vLLM
- 0.6B may accidentally "translate" instead of transcribing if the wrong language is forced
- vLLM is required for streaming — transformers only supports batch
- FlashAttention 2 needs to be compiled from source on aarch64

## Issues Found

1. **Venv shebang problem** — Same issue as Parakeet. `cp -a` of a venv leaves shebangs pointing to the original path
2. **qwen-asr installed in the wrong venv** — pip with an incorrect shebang installs packages in the original venv. Fix: correct shebangs BEFORE running `pip install`
3. **accelerate version conflict** — qwen-asr wants 1.12.0 but NeMo sets 1.13.0. Works with 1.13.0 despite the warning

## Configuration

```python
from qwen_asr import Qwen3ASRModel
import torch

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    max_inference_batch_size=1,
    max_new_tokens=256,
)

results = model.transcribe(
    audio=(audio_np, 16000),  # float32, 16kHz
    language=None,             # auto-detect
)
text = results[0].text
lang = results[0].language
```

## Installation on Jetson Thor

Qwen3-ASR runs as a RealtimeVoiceChat instance on port **8005**. The venv is cloned from the base project to preserve native Jetson libs.

```bash
# 1. Copy the base project
cp -r ~/workspace/realtimevoicechat ~/workspace/realtimevoicechat-qwen3asr

# 2. Copy the working venv
cp -a ~/workspace/realtimevoicechat/venv ~/workspace/realtimevoicechat-qwen3asr/venv

# 3. Fix shebangs (CRITICAL)
grep -rl "realtimevoicechat/venv/bin/python" ~/workspace/realtimevoicechat-qwen3asr/venv/bin/ | \
  xargs -I{} sed -i "1s|realtimevoicechat/venv|realtimevoicechat-qwen3asr/venv|" {}

# 4. Verify
head -1 ~/workspace/realtimevoicechat-qwen3asr/venv/bin/pip

# 5. Install qwen-asr
~/workspace/realtimevoicechat-qwen3asr/venv/bin/pip install qwen-asr

# 6. Verify installation location
~/workspace/realtimevoicechat-qwen3asr/venv/bin/pip show qwen-asr | grep Location

# 7. Place adapter files in code/
# Copy qwen3_asr_adapter/ directory into ~/workspace/realtimevoicechat-qwen3asr/code/

# 8. Edit server.py — change port to 8005
sed -i 's/port=8000/port=8005/' ~/workspace/realtimevoicechat-qwen3asr/code/server.py

# 9. Edit transcribe.py — add USE_QWEN3ASR conditional import
# Add:
#   USE_QWEN3ASR = os.environ.get("USE_QWEN3ASR", "").strip() == "1"
#   if USE_QWEN3ASR:
#       from qwen3_asr_adapter import AudioToTextRecorderQwen3ASR as AudioToTextRecorder
#       AudioToTextRecorderClient = AudioToTextRecorder

# 10. Create systemd service
sudo tee /etc/systemd/system/qwen3asr-voice.service << 'EOF'
[Unit]
Description=Qwen3-ASR Voice Chat (port 8005)
After=network.target

[Service]
Type=simple
User=bujosa
WorkingDirectory=/home/bujosa/workspace/realtimevoicechat-qwen3asr/code
Environment=PATH=/home/bujosa/workspace/realtimevoicechat-qwen3asr/venv/bin:/usr/local/bin:/usr/bin
Environment=USE_QWEN3ASR=1
ExecStart=/home/bujosa/workspace/realtimevoicechat-qwen3asr/venv/bin/python3 server.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now qwen3asr-voice

# 11. Verify
sudo journalctl -u qwen3asr-voice -f
# Wait for "Uvicorn running on http://0.0.0.0:8005"
# Model loads in ~126 seconds (downloads ~3.5GB on first run)
```

## Verdict

Best accuracy available in open-source. Use 1.7B on hardware with enough VRAM (4GB+). For Thor it is the obvious choice vs the 0.6B.
