---
 <p align="center">
    <img src="pic.jpg" width="400"/>
<p>

<p align="center">
        💜 <a href="https://github.com/sherif1313/3arab-TTS/"><b>Github</b></a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/sherif1313/">Hugging Face</a>&nbsp&nbsp |  &nbsp&nbsp📚 <a href="https://github.com/sherif1313/Arabic-English-handwritten-OCR-v3/tree/main">Cookbooks</a>&nbsp&nbsp 
<br>
🖥️ <a href="[https://huggingface.co/spaces/sherif1313/3arab-TTS-VoiceDesign-v2]">Demo</a>&nbsp&nbsp </a>
</p>

## 🌍 3arab-TTS  

An independent Arabic Text-to-Speech (TTS) model based on the **Rectified Flow Diffusion Transformer (RF-DiT)** architecture.with Voice Design capabilities for controllable speaker identity, pitch, and style.Instead of requiring reference audio for voice cloning, this model features Voice Design 70 different voices

The acoustic model was trained entirely from scratch on Arabic speech data using random initialization, with independently developed training and inference pipelines.


Installation

## Installation

```bash
git clone https://github.com/sherif1313/3arab-TTS.git
cd 3arab-TTS
uv sync --extra cu128`
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


 ##  🤝 Community Contributions Welcome

Contributions are highly appreciated, including:

    Arabic speech datasets
    training improvements
    inference optimizations
    bug fixes
    evaluation & testing
    documentation improvements

## Evaluation

The model has been evaluated primarily through human listening tests and qualitative assessment.

Current strengths:

- Natural Arabic pronunciation
- Good speaker consistency
- High-fidelity DACVAE latent reconstruction
- Strong performance on Modern Standard Arabic

## Data Annotation and Caption Generation

The initial voice captions and metadata annotations were generated using Qwen3-Omni-30B-A3B-Instruct and subsequently reviewed, filtered, refined, and integrated into the training pipeline.

The generated annotations provide detailed descriptions of:

- Speaker characteristics
- Vocal timbre and pitch
- Speaking style
- Prosody and rhythm
- Emotional tone
- Religious recitation styles
- Dialectal speech patterns

Particular emphasis was placed on Arabic speech, including both Modern Standard Arabic (MSA) and Egyptian Colloquial Arabic.

The resulting caption set provides broad coverage of Arabic speaking styles, narration, preaching, news delivery, poetry recitation, religious recitation, conversational speech, and Egyptian dialect speech patterns.

Compared to previous versions, support for Egyptian Colloquial Arabic has improved significantly in terms of pronunciation accuracy, fluency, naturalness, and conversational expressiveness.

## Religious Recitation and Tajweed

The training data contains a substantial amount of religious and Quranic content.

Special attention was given to captions describing:

- Tajweed-inspired recitation styles
- Reverent and solemn delivery
- Religious narration
- Sermon-like speaking styles

As a result, the model demonstrates improved handling of religious vocabulary, Quranic verses, Islamic terminology, and recitation-oriented speech patterns.

While the model is not intended to replace dedicated Quran recitation systems, it is capable of generating speech with religious intonation and recitation-like characteristics when guided through appropriate voice captions.

## Unique Contributions

To the best of our knowledge, this is among the first openly released Arabic TTS models to combine:

- Caption-based voice design
- Fine-grained voice description control
- Egyptian Colloquial Arabic support
- Religious recitation-oriented speech styles
- Arabic voice generation without requiring reference audio
- Large-scale Arabic speech training at 48 kHz

The model demonstrates strong performance across both Modern Standard Arabic and Egyptian Colloquial Arabic, while enabling controllable speaker identity, style, pitch, and delivery through natural-language voice descriptions.
## Egyptian Arabic Support

One of the major strengths of this model is its support for Egyptian Colloquial Arabic. Compared to earlier versions, the model demonstrates significantly improved pronunciation, fluency, and naturalness when generating Egyptian dialect speech, making it suitable for conversational and everyday spoken Arabic applications With the appropriate speaker used for that


## Novel Contributions

To the best of our knowledge, this is among the first openly released Arabic voice-design-oriented TTS models trained at this scale with caption-based voice control, combining:

- Detailed voice descriptions and speaker attributes.
- Egyptian Colloquial Arabic support.
- Religious and recitation-oriented speech styles.
- Caption-guided voice generation.
- Multi-style Arabic speech synthesis.

The model represents an important step toward controllable Arabic speech synthesis and expressive Arabic voice generation.

```bash
# ======================= قوائم الأنماط الصوتية =======================

MALE_CAPTIONS = [
    "يحدث بإيجاز ووضوح وحزم وبلهجة جادة. معدل الكلام سريع والتنغيم ثابت",
    "لهجة جادة ويتحدث بأسلوب السرد البطيء والثقيل. أسلوب مسرحي في النطاق المتوسط والمنخفض",
    "هادئ. تخلص من الانفعالات بلهجة واقعية",
    "هادئ ومتوسط إلى منخفض. يتحدث ببطء نوعًا ما وبلهجة سرد ثقيلة",
    "صوت عالٍ بسرعة وبهدوء، مثل المذيع",
    "بنبرة جادة وصادقة وسلسة، مثل السرد",
    "أسلوب يشبه السرد يتم من خلاله سرد القصة ببطء",
    "نغمة سريعة بعض الشيء وصوت واضح",
    "هادئ ومنخفض النبرة. يتحدث ببطء، ويفصل بين كل كلمة",
    "بوتيرة بطيئة ، مملوءًا بالحزن. يقمع تقلباته العاطفية ويروي بنبرة ثابتة",
    "تحدث بوتيرة سريعة إلى حد ما. لهجة جادة",
    "تحدث بنبرة جادة وبنبرة ثابتة يقمع عواطفه ويقرأ بصوت عالٍ بنبرة ثابتة",
    "تلاوة بطيئة نوعاً ما، كأنها تصلي صلاة شكر. إطالة نهاية الكلمة وخلق جو مقدس",
    "يقرأ بصوت عالٍ بصوت قوي متوسط المدى في جو مهيب وجدي",
    "يتحدث ببطء ورسمية، كما لو كان يصلي. أسلوب قراءة جاد وثقل",
    "قراءة بأسلوب السرد هادئة مع نغمة منخفضة وبسرعة بطيئة قليلاً. جو غامض",
    "تحدث بنبرة جادة و صوت ثابت ، مثل الراوي",
    "صوت عميق . يتم تلاوتها ببطء في جو مهيب وهادئ",
    "هادئ ومنخفض. يتحدث بسرعة ثابتة وبطيئة بعض الشيء",
    "أسلوب يشبه السرد يتم من خلاله شرح القصة بطريقة واقعية وبسرعة ثابتة وبطيئة قليلاً",
    "صوت عميق . يتحدث بنبرة سريعة وقوية إلى حد ما، مليئة بالغضب والكراهية",
    "مثل المذيع، فهو يتحدث بوضوح وواقعية",
    "يتحدث بلهجة وعظية. وله أسلوب تلاوة مقنع وأسلوب حديث واضح",
    "بنبرة واضحة. صوت جاد وحازم مثل الخطبة",
    "يقرأ الحقائق أسلوب السرد بطيء بعض الشيء بصوت عالٍ بنبرة مسطحة",
    "يتحدث بهدوء وبنبرة وعظية جادة. سرعة التحدث طبيعية، والتنغيم متواضع",
    "صوت هادئ ولطيف. تحدث وبامتنان. جو هادئ تحدث بأدب",
    "صوت واضحة مع إيقاع ثابت وسريع قليلاً. تحدث مثل إعلان",
    "تحدث بحزم وبسرعة وبلهجة قوية",
    "بنبرة سريعة وسلسة إلى حد ما. نغمة قراءة جادة بصوت متوسط المدى",
    "ينطق بوضوح، بلهجة تشبه الصوت الذي يحذر",
    "صوت عميق وهادئ مع جو جدي . أسلوب قراءة مسطح وثقيل",
    "يتحدث بشكل واقعي بنبرة ثابتة",
    "يتحدث بنبرة جدية، مثل الراوي. هناك القليل من التنغيم",
    "إيقاع سريع إلى حد ما وتفسيرات واضحة. الصوت واضح",
    "صوت قوي ولسان ناعم، ونبرة سريعة إلى حد ما",
    "يتحدث بصوت قوي وحازم مثل الرواية العربية. أجواء جدية ومهيبه",
    "صوت واضح وأسلوب جاد يشرح الحقائق بطريقة واقعية",
    "صوت عالٍ بوتيرة ثابتة، بنبرة مسطحة خالية من المشاعر",
    "وتيرة ثابتة وسريعة إلى حد ما. يُقرأ الكتاب بأسلوب سطحي يكبت العاطفة، ويعطي انطباعًا موضوعيًا.",
    "هادئ ومنخفض. مثل المذيع، فهو يذكر الحقائق بنبرة بطيئة ومسطحة",
    "نغمة صوتية هادئة وواقعية. نغمة صوته مسطحة ويتحدث بنبرة هادئة متوسطة منخفضة",
    "يتحدث بجدية مع صوت جهير متوسط الرنانة في جو مهيب. هناك القليل من التنغيم",
    "هادئ. ينطق كل كلمة بوضوح ويقرأ بصوت عالٍ ببطء وبشكل رسمي. أسلوب مثل السرد الجاد",
    "هادئة. لديه القليل من التقلبات العاطفية، ويتحدث بشكل واقعي بنبرة ثابتة",
    "صوت عميق . قراءة ببطء وجدية في جو مهيب",
    "النغمة بطيئة وخطيرة، مثل السرد، مما يخلق جوًا مهيبًا",
    "عميق وبارد. يتحدث بسرعة إلى حد ما وبلهجة مسطحة رافضة",
    "هادئ ومنخفض. قم بالرد بهدوء بنبرة هادئة تستبعد العاطفة",
    "النبرة بطيئة ومهذبة، ولها جو فكري أشبه بالسرد",
    "يقرأ الحقائق بصوت عالٍ بنبرة سلسة وسريعة إلى حد ما. جودة صوت واضحة وسهلة السمع",
    "منخفض وهادئ. سرعة بطيئة قليلاً ونبرة قوية وحازمة. يبدو الأمر وكأنه رواية",
    "نبره اخباريه",
    "تجويد",
    "الهدوء والحزن, أسلوب القراءة",
    "عاطفي،حزين,مناجاة،همس،قراءة الشعر",
    "غاضب، غير راض, محادثة",
    "جدي, أسلوب السرد",
]

FEMALE_CAPTIONS = [
    "صوت أنثوي. تحدث بوتيرة ثابتة، مثل السرد",
    "امرأة شابة. يتحدث بنبرة جدية",
    "امرأة شابة. تتحدث بنبرة بطيئة وهادئة",
    "صوت امرأة شابة. قوي وواثق",
    "صوت امرأة شابة. تتحدث بنبرة هادئة",
    "صوت امرأة شابة. تتحدث بنبرة هادئة وموضوعية",
    "صوت امرأة شابة. تتحدث بطريقة جادة",
    "صوت امرأة شابة . تتحدث بصراحة وبصوت عميق ومسطح",
    "صوت امرأة شابة . تتحدث بشكل واقعي",
    "صوت امرأة شابة . يتحدث بصوت قوي وواثق",
    "صوت أنثوي . جادة ونطقها واضح وسهل السمع",
    "صوت أنثوي . يتحدث بطريقة جدية ووقورة",
    "صوت أنثوي . يتحدث بسلاسة، مثل السرد",
    "صوت امرأة شابة. تتحدث بلهجة مصرية",
    "صوت انثوي نبره اخباريه",
    "صوت امرأة شابة. دون انفعال. نغمة مسطحة مثل السرد",
]
  ```

  ## Current Limitations

* Future versions may support these behaviors through additional training on expressive and non-verbal speech datasets containing laughter, crying, sighing, whispering, shouting, breathing sounds, and other paralinguistic vocal events.  

* Arabic Only: This model currently supports arabic text input only.

* Arabic numerals are reliably supported for simple numbers (e.g., 1–10). Larger numbers may produce better results when written in words rather than digits. For example, writing "ثمانمائة وخمسة وأربعون" may yield more accurate pronunciation than "845".


* Some Arabic words may require explicit diacritics (Tashkeel) to ensure correct pronunciation, especially when multiple readings are possible. For example:

  * Without diacritics: "ويستغفرون للذين أمنوا"
  * Preferred: "ويستغفرون للذين آمَنُوا"

* Pronunciation quality may vary for rare words, names, religious texts, and highly ambiguous unvocalized Arabic sentences.

* For better use removal of unnecessary symbols such as "-" "_" ","  "." or other non-linguistic characters.

* Generated speech quality is influenced by both input text and inference configuration. Achieving the best results may require careful tuning of generation parameters, including Seed, Number of Steps, guidance strength, and reference conditioning settings. Different voices, speaking styles, and text types may benefit from different parameter configurations.


# 📌 Project Overview

This project is a diffusion-based Arabic Text-to-Speech system inspired by modern latent-space speech synthesis architectures.
Inspired by:
- `Echo-TTS`
- `Irodori-TTS`




# 🏗️ Architecture

Instead of relying on discrete audio tokens common in traditional TTS systems, this model generates **Continuous Latent Representations** using `DACVAE`.

| Component | Description |
|---|---|
| Component | Description |
|------------|------------|
| RF-DiT | 12-layer diffusion transformer with 20 attention heads |
| DACVAE | Continuous latent audio codec (32-dim latent space) |
| Arabic Text Encoder | 512-dimensional transformer encoder (10 layers, 8 heads) |
| Reference Speaker Encoder | 768-dimensional transformer encoder (8 layers, 12 heads) |
| Continuous Latent Space | Preserves fine acoustic details and minimizes spectral distortion |
---

# 🌊 Continuous Latent Space

The system converts audio into compact continuous latent vectors (`32-dim`), which the diffusion model then learns to generate directly. This approach enables:
- ✅ Smoother temporal generation
- ✅ Reduced quantization artifacts
- ✅ Preservation of fine acoustic details (breathing, vocal characteristics, prosody)
- ✅ Improved stability for longer utterances

---

# 🎛️ Style & Pitch Control

The `RF-DiT` architecture supports conditional style embedding, allowing control over:
- Speaker identity & pitch/timbre
- Speech rate & rhythm
- Expressive characteristics  
*(Based on inference settings )*
  

# 🚀 Roadmap & Upcoming Updates

| Feature | Planned Updates |
|---|---|
| **Speakers** | Expand support to a larger pool of male & female speakers |
| **Training Data** | Scale to ~1000–2000 hours of high-quality Arabic speech |
| **Quality & Stability** | Improve pronunciation accuracy & reduce spectral artifacts |
| **Voice Cloning** | Experimental research toward zero-shot voice adaptation using 3–12 second reference audio |
| **Expressivity** | Integration of fine-grained emotional & stylistic controls |
د
## Training Data Composition

Approximate distribution:

- Modern Standard Arabic
- Egyptian Colloquial Arabic
- Religious content
- News and narration
- Conversational speech
- Male speakers
- Female speakers

## Integrated watermarking

The integrated SilentCipher technology allows for the direct application of strong, invisible audio watermarks to the generated output, without reducing sound quality.

## Voice Design

Unlike traditional Arabic TTS systems that require reference audio, the model can generate different speaker styles directly from natural-language voice descriptions without providing a reference recording.
 
Due to the limited availability of large-scale open Arabic speech datasets, a significant portion of the training data was collected from publicly available Arabic content and carefully filtered for quality.

The current release  include integrated audio watermarking. SilentCipher watermarking  inference releases without affecting audio quality.

## Arabic TTS Data Preparation Pipeline

Training Arabic TTS models is challenging due to limited data availability. Audio is often collected from sources like YouTube, focusing on clear speech such as news or audiobooks. The audio is then segmented into short clips (3–12 seconds) using snakers4/silero-vad, and transcribed into text with MohamedRashad/Arabic-Whisper-CodeSwitching-Edition. Finally, transcripts are reviewed and paired with audio to create a clean dataset for training.

## The current release demonstrates that open-source Arabic TTS systems can achieve a level of quality and naturalness comparable to many production-grade solutions. With over 700 hours of carefully curated Arabic speech and a large-scale RF-DiT architecture, 3arab-TTS establishes a strong baseline for next-generation Arabic speech synthesis.


## 🙏 Acknowledgments 

    Aratako/Irodori-TTS
    jordand/echo-tts-base
    LlamaForCausalLM
    facebook/dacvae-watermarked (Audio latent encoder)
    Sony/SilentCipher

All model training, pipeline implementation, and acoustic model weights were developed independently and trained from scratch.
No proprietary acoustic models, private datasets, or closed-source training pipelines were used during development.
## 📜 License

Licensed under the Apache 2.0 License.

