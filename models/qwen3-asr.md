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

## Verdict

Best accuracy available in open-source. Use 1.7B on hardware with enough VRAM (4GB+). For Thor it is the obvious choice vs the 0.6B.
