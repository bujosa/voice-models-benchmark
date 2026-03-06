# Complete Installation Guide — Jetson Thor

Step-by-step commands for deploying all three voice models on Jetson Thor. These commands are battle-tested from actual deployment sessions.

## Prerequisites

- Jetson Thor with JetPack 7.0, CUDA 13.0, Python 3.12
- A working `~/workspace/voice-chat/` project with a functional venv (for Parakeet and Qwen3-ASR)
- SSH access to the Jetson

## Important Notes

- **Never create a fresh venv** for Parakeet or Qwen3-ASR. The PyTorch wheel on the Jetson index does not include all native NVIDIA libs. Copy an existing working venv instead.
- **Always fix shebangs** after copying a venv with `cp -a`. Without this, `pip install` silently installs packages into the original venv.
- **pip nvidia-cublas breaks cuBLAS on Jetson.** After installing packages that pull nvidia-cublas, you must re-symlink to the JetPack system libs.

---

## 1. Parakeet-TDT 0.6B v3 (port 8004)

Uses the voice-chat framework with a custom adapter. Runtime: NeMo.

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

---

## 2. Qwen3-ASR 1.7B (port 8005)

Uses the voice-chat framework with a custom adapter. Runtime: qwen-asr (transformers).

```bash
# 1. Copy the base project
cp -r ~/workspace/voice-chat ~/workspace/voice-chat-qwen3asr

# 2. Copy the working venv
cp -a ~/workspace/voice-chat/venv ~/workspace/voice-chat-qwen3asr/venv

# 3. Fix shebangs (CRITICAL)
grep -rl "voice-chat/venv/bin/python" ~/workspace/voice-chat-qwen3asr/venv/bin/ | \
  xargs -I{} sed -i "1s|voice-chat/venv|voice-chat-qwen3asr/venv|" {}

# 4. Verify
head -1 ~/workspace/voice-chat-qwen3asr/venv/bin/pip

# 5. Install qwen-asr
~/workspace/voice-chat-qwen3asr/venv/bin/pip install qwen-asr

# 6. Verify installation location
~/workspace/voice-chat-qwen3asr/venv/bin/pip show qwen-asr | grep Location

# 7. Place adapter files in code/
# Copy qwen3_asr_adapter/ directory into ~/workspace/voice-chat-qwen3asr/code/

# 8. Edit server.py — change port to 8005
sed -i 's/port=8000/port=8005/' ~/workspace/voice-chat-qwen3asr/code/server.py

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
User=<your-user>
WorkingDirectory=/home/<your-user>/workspace/voice-chat-qwen3asr/code
Environment=PATH=/home/<your-user>/workspace/voice-chat-qwen3asr/venv/bin:/usr/local/bin:/usr/bin
Environment=USE_QWEN3ASR=1
ExecStart=/home/<your-user>/workspace/voice-chat-qwen3asr/venv/bin/python3 server.py
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

---

## 3. Qwen3-Omni 30B-A3B (port 8006)

This one does **NOT** use the voice-chat framework. It has its own standalone FastAPI server because the WebSocket protocol is incompatible.

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
# (see models/qwen3-omni.md Configuration section for the model loading code)

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

---

## 4. DNS and HTTPS Setup

Required for accessing the services via local domains with HTTPS from the local network.

### DNS entries (Local DNS Server)

```bash
# 1. Add DNS entries in your local DNS server (e.g., Pi-hole, dnsmasq, etc.)
# Point each domain to the IP of your reverse proxy server:
#   <proxy-ip> parakeet.local
#   <proxy-ip> qwen3-asr.local
#   <proxy-ip> qwen3-omni.local
# Then restart DNS (e.g., pihole restartdns)
```

### HTTPS certificates (with mkcert)

```bash
# 2. Generate HTTPS certs
mkcert parakeet.local
mkcert qwen3-asr.local
mkcert qwen3-omni.local

# 3. Upload certs to NPM (Nginx Proxy Manager)
# Place fullchain.pem + privkey.pem in NPM's /data/custom_ssl/npm-{N}/
```

### Nginx Proxy Manager reverse proxy

```bash
# 4. Create NPM proxy host configs
# NOTE: NPM has a bug where API-created hosts don't generate nginx configs.
# You must manually create config files in /data/nginx/proxy_host/
# Each config proxies HTTPS → http://<jetson-ip>:<port> with WebSocket support
```

---

## Port Summary

| Port | Model | Service Name | Type |
|------|-------|-------------|------|
| 8004 | Parakeet-TDT 0.6B v3 | parakeet-voice | voice-chat + adapter |
| 8005 | Qwen3-ASR 1.7B | qwen3asr-voice | voice-chat + adapter |
| 8006 | Qwen3-Omni 30B-A3B | qwen3omni-voice | Standalone FastAPI |

## Service Management

```bash
# Check status of all three services
for s in parakeet-voice qwen3asr-voice qwen3omni-voice; do
  echo -n "$s: "; systemctl is-active $s
done

# View logs
sudo journalctl -u parakeet-voice -f
sudo journalctl -u qwen3asr-voice -f
sudo journalctl -u qwen3omni-voice -f

# Stop/start individual services
sudo systemctl stop qwen3omni-voice
sudo systemctl start qwen3omni-voice

# Check RAM usage
free -h
```
