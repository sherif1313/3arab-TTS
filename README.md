# 3arab-tts
Arabic Text-to-Speech (RF-DiT Architecture)

This is a standalone implementation of the Modified Diffuse Transform (RF-DiT) algorithm for Arabic text-to-speech conversion.

It was trained from scratch on a custom Arabic dataset using random weight initialization.

## 📜 Inspiration and Architecture
The architecture is inspired by modern diffusion-based text-to-speech models, such as Echo-TTS and Irodori-TTS.

This model is a standalone implementation of an RF-DiT-based text-to-speech system, inspired by modern diffusion-based text-to-speech architectures, such as Echo-TTS and Irodori-TTS.


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
  --hf-checkpoint sherif1313/3arab-TTS-500M-V1 \
  --text "فسبحان الذي بيده ملكوت كل شيء وإليه ترجعون" \
  --no-ref \
  --output-wav outputs/sample.wav
  ```
