# Whisper (OpenAI)

## Tested Variants

| Variant | Parameters | Languages | Device |
|---------|-----------|---------|--------|
| base.en | 74M | English only | CPU |
| small | 244M | 99 languages | CPU |

## Advantages

- Mature ecosystem (RealtimeSTT, faster-whisper)
- Runs on CPU without GPU
- base.en is very lightweight in RAM (~1 GB)
- Well documented, easy to integrate
- Partial streaming with RealtimeSTT

## Disadvantages

- **base.en does not understand Spanish at all** — it made up words
- **small on CPU is slow** — noticeable latency vs GPU models
- No reliable auto language detection in base.en
- .en models are hardcoded to English, useless for multilingual
- English with Latin accent: frequent errors in base.en, acceptable in small
- Constantly consumes CPU during VAD + transcription
- High beam_size (3) worsens latency without significantly improving accuracy

## Optimal Configuration (tested on Thor)

```python
"model": "small",              # Do NOT use base.en for multilingual
"silero_sensitivity": 0.3,     # Default 0.05 cuts off phrases
"post_speech_silence_duration": 1.5,  # Default 0.7 is too aggressive
"beam_size": 1,                # 3 is unnecessarily slow
"realtime_processing_pause": 0.1,
```

## Verdict

Functional as a baseline but surpassed by everything else. Only makes sense if no GPU is available.
