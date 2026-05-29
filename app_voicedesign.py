#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
from huggingface_hub import snapshot_download

from arabic_tts.inference_runtime import (
    RuntimeKey,
    SamplingRequest,
    clear_cached_runtime,
    default_runtime_device,
    get_cached_runtime,
    list_available_runtime_devices,
    list_available_runtime_precisions,
    save_wav,
)
from arabic_tts.speaker_inversion import is_speaker_inversion_safetensors_path

# ==================== الثوابت المضافة ====================
CHECKPOINT = "sherif1313/3arab-TTS-500M-v1-VoiceDesign"
CODEC_REPO = "sherif1313/DACVAE-Arabic-32dim"
# ========================================================

MAX_GRADIO_CANDIDATES = 32

# قائمة الأنغام
TONE_CHOICES = [
    "نبرة طبيعية",
    "نبرة اخباريه",
    "نبرة هادئة",
    "صوت انثوي نبرة رسمية",
    "نبرة دينية",
    "نبرة رسمية هادئة",
    "صوت انثوي نبره اخباريه",

]

def _estimate_duration_from_text(text: str) -> float:
    words = text.split()
    word_count = len(words)
    estimated_seconds = (word_count / 2.0) + 1.5
    return max(2.0, estimated_seconds)

def _default_checkpoint() -> str: return CHECKPOINT
def _default_model_device() -> str: return default_runtime_device()
def _default_codec_device() -> str: return default_runtime_device()
def _precision_choices_for_device(device: str) -> list[str]: return list_available_runtime_precisions(device)
def _on_model_device_change(device: str) -> gr.Dropdown:
    choices = _precision_choices_for_device(device)
    return gr.Dropdown(choices=choices, value=choices[0])
def _on_codec_device_change(device: str) -> gr.Dropdown:
    choices = _precision_choices_for_device(device)
    return gr.Dropdown(choices=choices, value=choices[0])
def _on_t_schedule_mode_change(mode: str) -> object:
    return gr.update(interactive=str(mode).strip().lower() == "sway")

def _parse_optional_float(raw: str | None, label: str) -> float | None:
    if raw is None: return None
    text = str(raw).strip()
    if text == "" or text.lower() == "none": return None
    try: return float(text)
    except ValueError as exc: raise ValueError(f"{label} must be a float or blank.") from exc

def _parse_optional_int(raw: str | None, label: str) -> int | None:
    if raw is None: return None
    text = str(raw).strip()
    if text == "" or text.lower() == "none": return None
    try: return int(text)
    except ValueError as exc: raise ValueError(f"{label} must be an int or blank.") from exc

def _parse_optional_str(raw: str | None) -> str | None:
    if raw is None: return None
    text = str(raw).strip()
    if text == "" or text.lower() in {"none", "null", "off", "disable", "disabled", "base"}: return None
    return text

def _resolve_checkpoint_path(raw_checkpoint: str) -> str:
    checkpoint = str(raw_checkpoint).strip()
    if checkpoint == "": raise ValueError("checkpoint is required.")
    if is_speaker_inversion_safetensors_path(checkpoint): raise ValueError("Speaker embedding files cannot be used as model checkpoints.")
    path = Path(checkpoint)
    if path.is_file(): return str(path.resolve())
    try:
        local_dir = snapshot_download(repo_id=checkpoint)
        local_path = Path(local_dir)
        for ext in ["*.pt", "*.ckpt", "*.bin", "*.safetensors"]:
            found_files = list(local_path.glob(ext))
            if found_files: return str(found_files[0])
        raise FileNotFoundError(f"No model file found in repo: {checkpoint}")
    except Exception as e:
        raise FileNotFoundError(f"Failed to resolve checkpoint '{checkpoint}': {e}")

def _build_runtime_key(checkpoint: str, model_device: str, model_precision: str, codec_device: str, codec_precision: str) -> RuntimeKey:
    checkpoint_path = _resolve_checkpoint_path(checkpoint)
    return RuntimeKey(checkpoint=checkpoint_path, model_device=str(model_device), codec_repo=CODEC_REPO, model_precision=str(model_precision), codec_device=str(codec_device), codec_precision=str(codec_precision), compile_model=False, compile_dynamic=False)

def _generate_single_audio(runtime, text: str, caption: str, num_steps: int, seed: int | None, seconds: float | None, duration_scale: float, t_schedule_mode: str, sway_coeff: float, cfg_guidance_mode: str, cfg_scale_text: float, cfg_scale_caption: float, cfg_scale: float | None, cfg_min_t: float, cfg_max_t: float, context_kv_cache: bool, max_text_len: int | None, max_caption_len: int | None, truncation_factor: float | None, rescale_k: float | None, rescale_sigma: float | None, lora_adapter: str | None, log_fn) -> tuple[np.ndarray, int, dict]:
    actual_seconds = seconds
    if actual_seconds is None:
        actual_seconds = _estimate_duration_from_text(text)
        log_fn(f"[INFO] Auto-estimated duration: {actual_seconds:.2f}s")
    result = runtime.synthesize(SamplingRequest(text=text, caption=caption, ref_wav=None, ref_latent=None, no_ref=True, ref_normalize_db=-16.0, ref_ensure_max=True, num_candidates=1, decode_mode="sequential", seconds=actual_seconds, duration_scale=float(duration_scale), max_ref_seconds=30.0, max_text_len=max_text_len, max_caption_len=max_caption_len, num_steps=int(num_steps), seed=seed, cfg_guidance_mode=str(cfg_guidance_mode), cfg_scale_text=float(cfg_scale_text), cfg_scale_caption=float(cfg_scale_caption), cfg_scale_speaker=0.0, cfg_scale=cfg_scale, cfg_min_t=float(cfg_min_t), cfg_max_t=float(cfg_max_t), truncation_factor=truncation_factor, rescale_k=rescale_k, rescale_sigma=rescale_sigma, context_kv_cache=bool(context_kv_cache), speaker_kv_scale=None, speaker_kv_min_t=None, speaker_kv_max_layers=None, t_schedule_mode=str(t_schedule_mode), sway_coeff=float(sway_coeff), trim_tail=True, lora_adapter=lora_adapter), log_fn=log_fn)
    audio = result.audios[0].float().cpu().numpy()
    if audio.ndim == 2: audio = audio.mean(axis=0)
    audio = np.squeeze(audio)
    if audio.ndim == 0: audio = np.array([audio])
    return audio, result.sample_rate, {"seed_used": result.used_seed, "messages": result.messages}

def _concatenate_audios(audio_list: list[np.ndarray]) -> np.ndarray:
    if not audio_list: return np.array([])
    return np.concatenate([np.squeeze(a) for a in audio_list if a.size > 0])

def parse_dialogue_script(script: str, default_tone1: str, default_tone2: str, default_tone3: str) -> list[tuple[str, str]]:
    """يحلل النص ويربط النبرات بناءً على 1: و 2: و 3:"""
    lines = script.strip().split('\n')
    parsed = []
    speaker_index = 0  # 0 = متحدث 1, 1 = متحدث 2, 2 = متحدث 3
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        caption = None
        text = line
        
        # البحث عن النقطتين لمعرفة هل كتب المستخدم 1: أو 2: أو 3:
        if ':' in line:
            parts = line.split(':', 1)
            prefix = parts[0].strip()
            rest_of_text = parts[1].strip()
            
            if prefix == '1':
                caption = default_tone1
                text = rest_of_text
            elif prefix == '2':
                caption = default_tone2
                text = rest_of_text
            elif prefix == '3':
                caption = default_tone3
                text = rest_of_text
            # لو كتب اسم النبرة بالكامل
            elif prefix in TONE_CHOICES:
                caption = prefix
                text = rest_of_text
        
        # التبديل التلقائي إذا لم يكتب 1: أو 2: أو 3:
        if caption is None:
            if speaker_index % 3 == 0:
                caption = default_tone1
            elif speaker_index % 3 == 1:
                caption = default_tone2
            else:
                caption = default_tone3
            speaker_index += 1
            
        if text:
            parsed.append((caption, text))
            
    return parsed

def _run_dialogue_generation(
    checkpoint: str, model_device: str, model_precision: str, codec_device: str, codec_precision: str,
    dialogue_script: str, tone1: str, tone2: str, tone3: str,
    num_steps: int, seed_raw: str, seconds_raw: str, duration_scale: float,
    t_schedule_mode: str, sway_coeff: float, cfg_guidance_mode: str,
    cfg_scale_text: float, cfg_scale_caption: float, cfg_scale_raw: str,
    cfg_min_t: float, cfg_max_t: float, context_kv_cache: bool,
    max_text_len_raw: str, max_caption_len_raw: str, truncation_factor_raw: str,
    rescale_k_raw: str, rescale_sigma_raw: str, lora_adapter_raw: str,
) -> tuple[str | None, str, str]:
    
    def stdout_log(msg: str) -> None: print(msg, flush=True)

    parsed_lines = parse_dialogue_script(dialogue_script, tone1, tone2, tone3)
    if not parsed_lines:
        raise ValueError("لم يتم العثور على حوار صالح. استخدم التنسيق: 1: النص أو 2: النص")

    runtime_key = _build_runtime_key(checkpoint, model_device, model_precision, codec_device, codec_precision)
    runtime, reloaded = get_cached_runtime(runtime_key)
    if not runtime.model_cfg.use_caption_condition:
        raise ValueError("النموذج لا يدعم النبرات.")

    cfg_scale = _parse_optional_float(cfg_scale_raw, "cfg_scale")
    max_text_len = _parse_optional_int(max_text_len_raw, "max_text_len")
    max_caption_len = _parse_optional_int(max_caption_len_raw, "max_caption_len")
    truncation_factor = _parse_optional_float(truncation_factor_raw, "truncation_factor")
    rescale_k = _parse_optional_float(rescale_k_raw, "rescale_k")
    rescale_sigma = _parse_optional_float(rescale_sigma_raw, "rescale_sigma")
    seed = _parse_optional_int(seed_raw, "seed")
    manual_seconds = _parse_optional_float(seconds_raw, "seconds")
    lora_adapter = _parse_optional_str(lora_adapter_raw)

    out_dir = Path("/tmp/gradio_outputs_dialogue")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    audios = []
    sample_rate = None
    logs = []

    for i, (caption, text) in enumerate(parsed_lines, 1):
        stdout_log(f"Generating Line {i}/{len(parsed_lines)}: [{caption}] '{text}'")
        try:
            audio, sr, info = _generate_single_audio(runtime=runtime, text=text, caption=caption, num_steps=num_steps, seed=seed, seconds=manual_seconds, duration_scale=duration_scale, t_schedule_mode=t_schedule_mode, sway_coeff=sway_coeff, cfg_guidance_mode=cfg_guidance_mode, cfg_scale_text=cfg_scale_text, cfg_scale_caption=cfg_scale_caption, cfg_scale=cfg_scale, cfg_min_t=cfg_min_t, cfg_max_t=cfg_max_t, context_kv_cache=context_kv_cache, max_text_len=max_text_len, max_caption_len=max_caption_len, truncation_factor=truncation_factor, rescale_k=rescale_k, rescale_sigma=rescale_sigma, lora_adapter=lora_adapter, log_fn=stdout_log)
            audios.append(audio)
            if sample_rate is None: sample_rate = sr
            logs.append(f"Line {i} [{caption}]: Success")
        except Exception as e:
            logs.append(f"Line {i}: Error - {str(e)}")

    merged_path = None
    if audios:
        merged_audio = _concatenate_audios(audios)
        merged_path = str(out_dir / f"dialogue_{stamp}.wav")
        sf.write(merged_path, merged_audio, sample_rate)

    detail_text = "\n".join(logs)
    return merged_path, detail_text, "Done"

def build_ui() -> gr.Blocks:
    default_checkpoint = _default_checkpoint()
    default_model_device = _default_model_device()
    default_codec_device = _default_codec_device()
    device_choices = list_available_runtime_devices()
    model_precision_choices = _precision_choices_for_device(default_model_device)
    codec_precision_choices = _precision_choices_for_device(default_codec_device)

    with gr.Blocks(title="arabic-TTS Dialogue") as demo:
        gr.Markdown("# حوار بين 3 متحدثين - Arabic TTS Dialogue")
        gr.Markdown("""
        **طريقة الاستخدام:**
        اكتب `1:` للمتحدث الأول، و `2:` للمتحدث الثاني، و `3:` للمتحدث الثالث.
        إذا أردت إعادة المتحدث الأول بعد الثالث، اكتب `1:` مرة أخرى.
        إذا لم تكتب رقماً، سيتم التبديل التلقائي (1 ثم 2 ثم 3 ثم 1...).
        """)

        with gr.Row():
            checkpoint = gr.Textbox(label="Model Checkpoint", value=default_checkpoint, scale=4, visible=False)
            model_device = gr.Dropdown(label="Model Device", choices=device_choices, value=default_model_device, scale=1)
            model_precision = gr.Dropdown(label="Model Precision", choices=model_precision_choices, value=model_precision_choices[0], scale=1)
            codec_device = gr.Dropdown(label="Codec Device", choices=device_choices, value=default_codec_device, scale=1)
            codec_precision = gr.Dropdown(label="Codec Precision", choices=codec_precision_choices, value=codec_precision_choices[0], scale=1)

        dialogue_script = gr.Textbox(
            label="نص الحوار", 
            lines=10, 
            placeholder="1: مرحباً، هل قرأتم الكتاب الجديد؟\n2: نعم، بدأته للتو.\n3: وأنا أيضاً، الفصل الأول مذهل.\n1: بالتأكيد، أنا أوافقكم الرأي.",
            value="1: هل انتهيت من قراءة هذا الكتاب؟\n2: ليس بالكامل، لكنني وصلت إلى الفصل الأخير.\n3: أنا سمعت عنه كثيراً، هل يستحق القراءة؟\n1: بالتأكيد، خاصة الجزء المتعلق بتطوير الذات.\n2: أوافقك الرأي، أسلوب الكاتب جميل."
        )

        with gr.Row():
            tone1 = gr.Dropdown(label="نبرة الرقم 1", choices=TONE_CHOICES, value="نبرة طبيعية")
            tone2 = gr.Dropdown(label="نبرة الرقم 2", choices=TONE_CHOICES, value="صوت انثوي نبرة رسمية")
            tone3 = gr.Dropdown(label="نبرة الرقم 3", choices=TONE_CHOICES, value="نبرة دينية")

        with gr.Accordion("إعدادات التوليد", open=True):
            with gr.Row():
                num_steps = gr.Slider(label="Num Steps", minimum=1, maximum=120, value=16, step=1)
                seed_raw = gr.Textbox(label="Seed", value="")
                seconds_raw = gr.Textbox(label="Seconds (blank=auto)", value="")
                duration_scale = gr.Slider(label="Duration Scale", minimum=0.5, maximum=1.5, value=1.0, step=0.01)
            with gr.Row():
                t_schedule_mode = gr.Dropdown(label="Time Schedule", choices=["linear", "sway"], value="linear")
                sway_coeff = gr.Slider(label="Sway Coeff", minimum=-1.0, maximum=1.5, value=-1.0, step=0.1, interactive=False)
            with gr.Row():
                cfg_guidance_mode = gr.Dropdown(label="CFG Guidance Mode", choices=["independent", "joint", "alternating"], value="independent")
                cfg_scale_text = gr.Slider(label="CFG Scale Text", minimum=0.0, maximum=10.0, value=2.0, step=0.1)
                cfg_scale_caption = gr.Slider(label="CFG Scale Caption", minimum=0.0, maximum=10.0, value=4.0, step=0.1)

        with gr.Accordion("خيارات متقدمة", open=False):
            cfg_scale_raw = gr.Textbox(label="CFG Scale Override (optional)", value="")
            with gr.Row():
                cfg_min_t = gr.Number(label="CFG Min t", value=0.5)
                cfg_max_t = gr.Number(label="CFG Max t", value=1.0)
                context_kv_cache = gr.Checkbox(label="Context KV Cache", value=True)
            with gr.Row():
                max_text_len_raw = gr.Textbox(label="Max Text Len (optional)", value="")
                max_caption_len_raw = gr.Textbox(label="Max Caption Len (optional)", value="")
            with gr.Row():
                truncation_factor_raw = gr.Textbox(label="Truncation Factor (optional)", value="")
                rescale_k_raw = gr.Textbox(label="Rescale k (optional)", value="")
                rescale_sigma_raw = gr.Textbox(label="Rescale sigma (optional)", value="")
            lora_adapter_raw = gr.Textbox(label="LoRA Adapter Directory (optional)", value="")

        generate_btn = gr.Button("إنشاء الحوار", variant="primary")

        out_audio_merged = gr.Audio(label="الحوار المدمج", type="filepath", interactive=False)
        out_log = gr.Textbox(label="سجل التشغيل", lines=6)
        out_timing = gr.Textbox(label="التوقيت", lines=4, visible=False)

        generate_btn.click(
            _run_dialogue_generation,
            inputs=[
                checkpoint, model_device, model_precision, codec_device, codec_precision, 
                dialogue_script, tone1, tone2, tone3, 
                num_steps, seed_raw, seconds_raw, duration_scale, t_schedule_mode, sway_coeff, cfg_guidance_mode, cfg_scale_text, cfg_scale_caption, cfg_scale_raw, cfg_min_t, cfg_max_t, context_kv_cache, max_text_len_raw, max_caption_len_raw, truncation_factor_raw, rescale_k_raw, rescale_sigma_raw, lora_adapter_raw
            ],
            outputs=[out_audio_merged, out_log, out_timing],
        )
        model_device.change(_on_model_device_change, inputs=[model_device], outputs=[model_precision])
        codec_device.change(_on_codec_device_change, inputs=[codec_device], outputs=[codec_precision])
        t_schedule_mode.change(_on_t_schedule_mode_change, inputs=[t_schedule_mode], outputs=[sway_coeff])

    return demo

def main() -> None:
    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == "__main__":
    main()
