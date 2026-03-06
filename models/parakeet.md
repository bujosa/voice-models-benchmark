# NVIDIA Parakeet-TDT 0.6B v3

## Specs

| Field | Value |
|-------|-------|
| Model | nvidia/parakeet-tdt-0.6b-v3 |
| Parameters | 600M |
| Architecture | FastConformer-TDT |
| Runtime | NeMo 2.7+ |
| Device | GPU |
| VRAM | ~2 GB |
| Languages | 25 (European + RU/UK) |
| License | CC-BY-4.0 |

## Advantages

- **The fastest STT in the world** — RTFx 3332 in benchmarks
- 25 languages including EN and ES with auto-detection
- Only 2 GB VRAM — fits alongside any other model
- Automatic punctuation and capitalization
- Word-level timestamps
- Supports long audio (up to 3 hours with local attention)
- Model loaded in 78s on Jetson Thor

## Disadvantages

- **NeMo is heavy** — installs ~200 packages, Megatron, PyTorch Lightning
- Megatron/OneLogger warnings on every startup (cosmetic but ugly)
- Streaming is not native — requires chunked inference which is not battle-tested for TDT
- Does not support Asian languages (no JA, KO, ZH)
- ONNX model discovery fails on Jetson (warning, not error)

## Issues Found

1. **libnvpl_lapack/blas/libcudss missing** — When copying a torch venv, NVIDIA libs are missing. Fix: manually copy `libnvpl_*.so`, `libarm_compute.so`, `libgfortran.so.5`, etc. from a working venv
2. **Incorrect shebangs** — When copying venvs with `cp -a`, the shebangs of pip/scripts point to the original path. Fix: `sed -i "1s|old/venv|new/venv|"` on all scripts in bin/
3. **pip installs in the wrong venv** — Consequence of the incorrect shebang. Always verify with `pip show <pkg> | grep Location`

## Configuration

```python
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
model.eval()
output = model.transcribe(audio=[audio_np])  # float32, 16kHz, mono
text = output[0].text
```

## Installation on Jetson Thor

Parakeet runs as a voice-chat instance on port **8004**. The venv is cloned from the base project to preserve native Jetson libs.

```bash
# 1. Copy the base voice-chat project
cp -r ~/workspace/voice-chat ~/workspace/voice-chat-parakeet

# 2. Copy the working venv (do NOT create fresh — native libs will be missing)
cp -a ~/workspace/voice-chat/venv ~/workspace/voice-chat-parakeet/venv

# 3. Fix shebangs (CRITICAL — without this, pip installs to the WRONG venv)
grep -rl "voice-chat/venv/bin/python" ~/workspace/voice-chat-parakeet/venv/bin/ | \
  xargs -I{} sed -i "1s|voice-chat/venv|voice-chat-parakeet/venv|" {}

# 4. Verify shebangs are correct
head -1 ~/workspace/voice-chat-parakeet/venv/bin/pip
# Should show: #!/home/<your-user>/workspace/voice-chat-parakeet/venv/bin/python3

# 5. Install NeMo (Parakeet's runtime)
~/workspace/voice-chat-parakeet/venv/bin/pip install nemo-toolkit[asr]

# 6. Verify it installed in the RIGHT venv
~/workspace/voice-chat-parakeet/venv/bin/pip show nemo-toolkit | grep Location
# Must show: /home/<your-user>/workspace/voice-chat-parakeet/venv/lib/...

# 7. Fix cuBLAS symlinks (NeMo pulls nvidia-cublas which breaks JetPack)
VENV_NVIDIA=~/workspace/voice-chat-parakeet/venv/lib/python3.12/site-packages/nvidia
SYSTEM_CUBLAS=/usr/local/cuda/lib64
ln -sf $SYSTEM_CUBLAS/libcublas.so.13 $VENV_NVIDIA/cublas/lib/libcublas.so.13
ln -sf $SYSTEM_CUBLAS/libcublasLt.so.13 $VENV_NVIDIA/cublas/lib/libcublasLt.so.13

# 8. Place adapter files in code/
# Copy parakeet_adapter/ directory into ~/workspace/voice-chat-parakeet/code/

# 9. Edit server.py — change port to 8004
sed -i 's/port=8000/port=8004/' ~/workspace/voice-chat-parakeet/code/server.py

# 10. Edit transcribe.py — add USE_PARAKEET conditional import at the top (after existing imports)
# Add:
#   USE_PARAKEET = os.environ.get("USE_PARAKEET", "").strip() == "1"
#   if USE_PARAKEET:
#       from parakeet_adapter import AudioToTextRecorderParakeet as AudioToTextRecorder
#       AudioToTextRecorderClient = AudioToTextRecorder

# 11. Create systemd service
sudo tee /etc/systemd/system/parakeet-voice.service << 'EOF'
[Unit]
Description=Parakeet Voice Chat (port 8004)
After=network.target

[Service]
Type=simple
User=<your-user>
WorkingDirectory=/home/<your-user>/workspace/voice-chat-parakeet/code
Environment=PATH=/home/<your-user>/workspace/voice-chat-parakeet/venv/bin:/usr/local/bin:/usr/bin
Environment=USE_PARAKEET=1
ExecStart=/home/<your-user>/workspace/voice-chat-parakeet/venv/bin/python3 server.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now parakeet-voice

# 12. Verify
sudo journalctl -u parakeet-voice -f
# Wait for "Uvicorn running on http://0.0.0.0:8004"
# Model loads in ~78 seconds
```

## Verdict

Best speed/accuracy/VRAM ratio. Recommended as the default STT for real-time voice applications.
