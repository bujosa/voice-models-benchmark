# Qwen3-TTS (Alibaba/Qwen)

## Specs

| Field | 0.6B | 1.7B |
|-------|------|------|
| Base Model | Qwen/Qwen3-TTS-12Hz-0.6B-Base | Qwen/Qwen3-TTS-12Hz-1.7B-Base |
| CustomVoice | Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice | Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice |
| VoiceDesign | — | Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign |
| VRAM | ~3 GB | ~6 GB |
| First chunk latency | 93ms (0.6B, RTX 4090) | 97ms (1.7B, RTX 4090) |
| Languages | 10 | 10 |
| Output sample rate | 24 kHz | 24 kHz |
| License | Apache 2.0 | Apache 2.0 |

## Three Modes

1. **Base** — Voice cloning from reference audio (min 3s)
2. **CustomVoice** — 9 predefined voices + emotion/prosody control via instructions
3. **VoiceDesign** — Describe the voice in natural language and the model generates it

## Advantages

- **97ms first chunk latency** — faster than most TTS
- Voice cloning with only 3 seconds of reference
- 10 languages including EN and ES
- Cross-lingual cloning (clone a voice in one language, synthesize in another)
- 9 predefined voices with emotion control
- VoiceDesign allows creating voices by describing characteristics
- WER < 1.3% in English (very natural)

## Disadvantages

- **No official streaming yet** — batch inference only
- Streaming available via community forks (dffdeeq/Qwen3-TTS-streaming)
- Requires a separate Tokenizer (Qwen3-TTS-Tokenizer-12Hz) — downloads automatically but adds VRAM
- `transformers==4.57.3` pin — may conflict with Qwen3-ASR (4.57.6)
- FlashAttention on aarch64 requires manual compilation
- Only 3 "tier 1" languages (ZH, EN, JA) — the rest have less training data
- Not yet tested on Jetson Thor (pending deployment)

## Comparison vs Kokoro (current TTS)

| Aspect | Kokoro | Qwen3-TTS 1.7B |
|--------|--------|----------------|
| Languages | EN only | 10 languages |
| VRAM | ~0.5 GB | ~6 GB |
| Voice cloning | No | Yes (3s reference) |
| Emotion control | No | Yes (instructions) |
| Latency | Fast | 97ms first chunk |
| EN quality | Good | Excellent |
| ES quality | N/A | Good |

## Configuration

```python
from qwen_tts import Qwen3TTSModel
import torch

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",  # or "flash_attention_2"
)

# Voice cloning
wavs, sr = model.generate_voice_clone(
    text="Hola, esto es una prueba.",
    language="Spanish",
    ref_audio="referencia.wav",
    ref_text="Texto del audio de referencia.",
)
# sr = 24000
```

## Verdict

Natural replacement for Kokoro when multilingual support or voice cloning is needed. Pending integration with the voice-chat pipeline.
