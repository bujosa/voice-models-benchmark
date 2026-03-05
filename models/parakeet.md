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
- Model loaded in 78s on Thor

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

## Verdict

Best speed/accuracy/VRAM ratio. Recommended as the default STT for real-time voice applications.
