# Services on Jetson Thor

## Current Status

| Port | Service | STT | LLM | TTS | Systemd |
|------|---------|-----|-----|-----|---------|
| 8000 | FRIDAY | Whisper base.en | gemma3:4b | Kokoro | friday.service |
| 8001 | Wednesday UI | — | — | — | wednesday.service |
| 8002 | Whisper Small | Whisper small | gemma3:4b | Kokoro | whisper-small.service |
| 8003 | Canary | Canary 180M | gemma3:4b | Kokoro | canary-voice.service |
| 8004 | Parakeet | Parakeet-TDT 0.6B | gemma3:4b | Kokoro | parakeet-voice.service |
| 8005 | Qwen3-ASR | Qwen3-ASR 1.7B | gemma3:4b | Kokoro | qwen3asr-voice.service |
| 8006 | Qwen3-Omni | Qwen3-Omni 30B (all-in-one) | — | — | qwen3omni-voice.service |

## Estimated RAM per Service

| Service | Direct RAM | Mapped RAM (cache) |
|---------|-----------|---------------------|
| FRIDAY | ~3 GB | ~1 GB |
| Whisper Small | ~3.5 GB | ~1.5 GB |
| Canary | ~3 GB | ~1 GB |
| Parakeet | ~4 GB | ~2 GB |
| Qwen3-ASR | ~5 GB | ~4 GB |
| Qwen3-Omni | ~27 GB | ~50 GB (mmap) |
| Ollama gemma3:4b | ~5 GB | ~4 GB |

**Jetson Thor total:** 122 GB unified RAM

## Useful Commands

```bash
# Check status of all services
for s in friday wednesday whisper-small canary-voice parakeet-voice qwen3asr-voice qwen3omni-voice; do
  echo -n "$s: "; systemctl is-active $s
done

# Stop services to free RAM
sudo systemctl stop whisper-small canary-voice

# View logs
sudo journalctl -u parakeet-voice -f

# Check RAM
free -h
```

## DNS (Local DNS Server)

All local domains point to the DNS/proxy server where NPM handles reverse proxy to Jetson Thor.

## HTTPS (NPM on Proxy Server)

Certificates generated with `mkcert` on a development machine. Root CA already trusted in the system keychain.

**Known bug:** NPM API creates DB entries for proxy hosts but does not generate nginx configs automatically. Workaround: create configs manually in `/data/nginx/proxy_host/`.
