# 3arab-tts
Arabic Text-to-Speech (RF-DiT Architecture)

This is a standalone implementation of the Modified Diffuse Transform (RF-DiT) algorithm for Arabic text-to-speech conversion.

It was trained from scratch on a custom Arabic dataset using random weight initialization.

## 📜 Inspiration and Architecture
The architecture is inspired by modern diffusion-based text-to-speech models.

This model is a standalone implementation of an RF-DiT-based text-to-speech system, inspired by modern diffusion-based text-to-speech architectures,  Echo-TTS and Irodori-TTS.


Installation

## Installation

```bash
git clone https://github.com/sherif1313/3arab-TTS.git
cd 3arab-TTS
uv sync
```

## Quick Start

```bash
uv run python infer.py \
  --hf-checkpoint sherif1313/3arab-TTS-500M-v2 \
  --text "فسبحان الذي بيده ملكوت كل شيء وإليه ترجعون" \
  --no-ref \
  --output-wav outputs/sample.wav
  ```

```bash
uv run python infer.py \
  --hf-checkpoint sherif1313/3arab-TTS-500M-v2 \
  --text "فسبحان الذي بيده ملكوت كل شيء وإليه ترجعون" \
  --ref-wav 5.wav \
  --output-wav outputs/sample.wav
  ```
## VoiceDesign Inference

```bash
uv run python infer.py \
  --hf-checkpoint sherif1313/3arab-TTS-500M-v2-VoiceDesign \
  --text "هذا السؤال وحده يمكن ان يغير حياتك بالكامل。" \
  --caption "يحدث بإيجاز ووضوح وحزم وبلهجة جادة" \
  --no-ref \
  --output-wav outputs/sample_voice_design.wav
```

## Web UI 

```bash
uv run python app.py
```

## Web UI VoiceDesign

```bash
uv run python app_voicedesign.py
```

## Web UI Dialogue

```bash
uv run python app_tts_dialogue.py
```

## Integrated watermarking

The integrated SilentCipher technology allows for the direct application of strong, invisible audio watermarks to the generated output, without reducing sound quality. These watermarks can be added to the playback code later.
```python

        _log(
            (
                "[runtime] start synthesize "
                "model_device={} model_precision={} codec_device={} codec_precision={} "
                "silentcipher_watermark={} mode={} seconds={} steps={} seed={} candidates={} decode_mode={}"
            ).format(
                self.key.model_device,
                self.key.model_precision,
                self.key.codec_device,
                self.key.codec_precision,
                self.watermarker.ready,
                req.cfg_guidance_mode,
                req.seconds,
                req.num_steps,
                "random" if req.seed is None else int(req.seed),
                req.num_candidates,
                req.decode_mode,
            )
        )
```

## Arabic TTS Data Preparation Pipeline

Training Arabic TTS models is challenging due to limited data availability. Audio is often collected from sources like YouTube, focusing on clear speech such as news or audiobooks. The audio is then segmented into short clips (3–12 seconds) using snakers4/silero-vad, and transcribed into text with MohamedRashad/Arabic-Whisper-CodeSwitching-Edition. Finally, transcripts are reviewed and paired with audio to create a clean dataset for training.
