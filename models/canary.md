# NVIDIA Canary 180M Flash

## Specs

| Field | Value |
|-------|-------|
| Model | istupakov/canary-180m-flash-onnx |
| Parameters | 180M |
| Runtime | ONNX (onnx-asr) |
| Device | GPU |
| VRAM | ~0.5 GB |

## Advantages

- Very lightweight in memory (180M params)
- Runs on GPU via ONNX — faster than Whisper on CPU
- Supports EN, ES, DE, FR
- Simple installation with `pip install onnx-asr`
- Good model for testing the adapter pattern

## Disadvantages

- **Only 4 languages** — limited vs Parakeet (25) or Qwen3 (30)
- Lower accuracy than Parakeet and Qwen3-ASR
- ONNX runtime on Jetson has GPU discovery warnings (works but ugly)
- The onnx-asr package has poor documentation
- No native streaming
- Does not auto-detect language — it must be specified

## Issues Found

1. **OnnxWrapper does not have `.eval()`** — Silero VAD with `onnx=True` returns a wrapper that does not support `.eval()`. Fix: `if hasattr(model, 'eval'): model.eval()`
2. **Conflict with `model` param** — RealtimeSTT passes `model: "base.en"` which Canary interpreted as its model ID. Fix: hardcode `CANARY_MODEL` as a class attribute and absorb via `**kwargs`
3. **cuBLAS symlinks** — pip nvidia-cublas breaks cuBLAS on Jetson. The system JetPack libs must be re-symlinked

## Verdict

Served as a proof of concept for the adapter pattern. Surpassed by Parakeet in speed and Qwen3-ASR in accuracy.
