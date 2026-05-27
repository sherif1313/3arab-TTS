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
  --hf-checkpoint sherif1313/3arab-TTS-500M-v1 \
  --text "فسبحان الذي بيده ملكوت كل شيء وإليه ترجعون" \
  --no-ref \
  --output-wav outputs/sample.wav
  ```

```bash
uv run python infer.py \
  --hf-checkpoint sherif1313/3arab-TTS-500M-v1 \
  --text "فسبحان الذي بيده ملكوت كل شيء وإليه ترجعون" \
  --ref-wav 5.wav \
  --output-wav outputs/sample.wav
  ```
## Web UI

```bash
uv run python app.py
```

## Arabic TTS Data Preparation Pipeline

Training Arabic TTS models is challenging due to limited data availability. Audio is often collected from sources like YouTube, focusing on clear speech such as news or audiobooks. The audio is then segmented into short clips (3–7 seconds) using snakers4/silero-vad, and transcribed into text with MohamedRashad/Arabic-Whisper-CodeSwitching-Edition. Finally, transcripts are reviewed and paired with audio to create a clean dataset for training.
