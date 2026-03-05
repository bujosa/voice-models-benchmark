# STT Adapter Pattern

## Concept

RealtimeVoiceChat uses `AudioToTextRecorder` from RealtimeSTT (based on Whisper). To use other STT engines without rewriting the entire framework, we created drop-in adapters that implement the same interface.

## Required Interface

```python
class AudioToTextRecorderAdapter:
    # Required attributes
    frames: collections.deque[bytes]    # raw frames buffer
    frames_lock: threading.Lock
    is_recording: bool

    def __init__(self, **kwargs):
        """Accepts all RealtimeSTT params via **kwargs, ignores irrelevant ones."""

    def feed_audio(self, chunk: bytes) -> None:
        """Receives PCM Int16 at 16kHz mono."""

    def text(self, on_transcription_finished=None) -> str:
        """Blocks until final transcription is available. Returns text."""

    def get_parameter(self, name: str) -> Any:
        """Reads parameters (especially 'is_recording')."""

    def set_parameter(self, name: str, value: Any) -> None:
        """Writes parameters."""

    def shutdown(self) -> None:
        """Cleanup."""
```

## Adapter Structure

```
<name>_adapter/
  __init__.py           # Exports the STT class and Recorder
  <name>_stt.py         # Model wrapper (transcribes float32 → str)
  recorder_adapter.py   # Implements the AudioToTextRecorder interface
```

## Recorder Components

1. **Silero VAD** — Detects voice vs silence (shared across all adapters)
2. **Process loop** — Accumulates audio, detects speech start/end
3. **Realtime loop** — Partial transcription every ~0.4s during speech
4. **Callbacks** — `on_recording_start/stop`, `on_realtime_transcription_update`, `on_turn_detection_start/stop`

## Activation via Environment Variable

```python
# In transcribe.py
import os
USE_PARAKEET = os.environ.get("USE_PARAKEET", "").strip() == "1"

if USE_PARAKEET:
    from parakeet_adapter import AudioToTextRecorderParakeet as AudioToTextRecorder
    AudioToTextRecorderClient = AudioToTextRecorder
elif START_STT_SERVER:
    from RealtimeSTT import AudioToTextRecorderClient
else:
    from RealtimeSTT import AudioToTextRecorder
```

```ini
# In systemd service
[Service]
Environment=USE_PARAKEET=1
```

## Implemented Adapters

| Adapter | Model | Env Var | File |
|---------|-------|---------|------|
| canary_adapter | Canary 180M Flash | USE_CANARY=1 | canary_stt.py |
| parakeet_adapter | Parakeet-TDT 0.6B | USE_PARAKEET=1 | parakeet_stt.py |
| qwen3_asr_adapter | Qwen3-ASR 1.7B | USE_QWEN3ASR=1 | qwen3_asr_stt.py |

## Common Errors

1. **`model` param conflict** — RealtimeSTT config includes `model: "base.en"`. If your adapter receives this as the first param, it fails. Solution: use `**kwargs` and hardcode the model ID as a class attribute.

2. **Silero VAD ONNX `.eval()`** — With `onnx=True`, Silero returns an OnnxWrapper that does not have `.eval()`. Guard with: `if hasattr(model, 'eval'): model.eval()`

3. **`frames` and `frames_lock` are mandatory** — The framework reads these attributes. If they don't exist, it causes a silent crash.
