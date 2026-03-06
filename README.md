# Voice Models Benchmark

Practical evaluation of STT, TTS, and Omni models deployed on edge hardware (Jetson Thor, 122GB unified RAM, Blackwell GPU, CUDA 13.0).

Results based on real-world tests, not paper benchmarks.

## Quick Summary

| Model | Type | VRAM | Speed | Spanish | Accented English | Verdict |
|-------|------|------|-------|---------|-------------------|---------|
| Whisper base.en | STT | ~1 GB | Fast (CPU) | ❌ No | ⚠️ Poor | Native English only |
| Whisper small | STT | ~2 GB | Medium (CPU) | ✅ Yes | ⚠️ Fair | Slight improvement, slow on CPU |
| Canary 180M Flash | STT | ~0.5 GB | Fast (GPU) | ✅ Yes | ⚠️ Fair | Lightweight but limited |
| Parakeet-TDT 0.6B | STT | ~2 GB | Very fast (GPU) | ✅ Yes | ✅ Good | Best speed/accuracy |
| Qwen3-ASR 1.7B | STT | ~4 GB | Fast (GPU) | ✅ Yes | ✅ Very good | Best open-source accuracy |
| Qwen3-TTS 1.7B | TTS | ~6 GB | 97ms first chunk | ✅ Yes | ✅ Yes | Kokoro replacement |
| Qwen3-Omni 30B | Omni | ~70 GB | ~500ms-1s | ✅ Yes | ✅ Excellent | Replaces STT+LLM+TTS |
| Kokoro | TTS | ~0.5 GB | Fast | ❌ EN only | ✅ Yes | Current, English only |

## Structure

```
models/
  whisper.md          # Whisper base.en and small
  canary.md           # NVIDIA Canary 180M Flash
  parakeet.md         # NVIDIA Parakeet-TDT 0.6B v3
  qwen3-asr.md        # Qwen3-ASR 0.6B and 1.7B
  qwen3-tts.md        # Qwen3-TTS 0.6B and 1.7B
  qwen3-omni.md       # Qwen3-Omni 30B-A3B
deployment/
  thor-services.md    # Active services on Jetson Thor
  adapters.md         # STT adapter pattern
  venv-issues.md      # Common venv issues on Jetson
```

## Conclusions

1. **For pure speed:** Parakeet-TDT — the fastest STT in existence, 25 languages
2. **For maximum STT accuracy:** Qwen3-ASR 1.7B — SOTA open-source, 30 languages
3. **To replace the entire pipeline:** Qwen3-Omni — a single model handles STT+reasoning+TTS
4. **For multilingual TTS:** Qwen3-TTS — 97ms latency, voice cloning in 3 seconds
5. **Avoid:** Whisper base.en for anything other than native English

## Current Stack

```
Microphone → WebSocket → Silero VAD → [STT Engine] → Ollama gemma3:4b → Kokoro TTS → Audio
```

With Qwen3-Omni the stack simplifies to:

```
Microphone → WebSocket → Qwen3-Omni → Audio
```
