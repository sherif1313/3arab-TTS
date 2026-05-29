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
GRADIO_AUDIO_COLS_PER_ROW = 8

# قائمة الأنغام
TONE_CHOICES = [
    "نبرة طبيعية",
    "نبرة اخباريه",
    "نبرة هادئة",
    "نبرة طبيعية هادئة",
    "صوت انثوي نبرة رسمية",
    "نبرة دينية",
    "نبرة رسمية",
    "نبرة رسمية هادئة",
    "نبره اخباريه حاده",
    "صوت انثوي نبره اخباريه",
    "نبره حاده"
]

def _estimate_duration_from_text(text: str) -> float:
    """تقدر مدة الصوت بالثواني بناءً على عدد كلمات النص العربي"""
    words = text.split()
    word_count = len(words)
    estimated_seconds = (word_count / 2.0) + 1.5
    return max(2.0, estimated_seconds)

def _default_checkpoint() -> str:
    return CHECKPOINT

def _default_model_device() -> str:
    return default_runtime_device()

def _default_codec_device() -> str:
    return default_runtime_device()

def _precision_choices_for_device(device: str) -> list[str]:
    return list_available_runtime_precisions(device)

def _on_model_device_change(device: str) -> gr.Dropdown:
    choices = _precision_choices_for_device(device)
    return gr.Dropdown(choices=choices, value=choices[0])

def _on_codec_device_change(device: str) -> gr.Dropdown:
    choices = _precision_choices_for_device(device)
    return gr.Dropdown(choices=choices, value=choices[0])

def _on_t_schedule_mode_change(mode: str) -> object:
    return gr.update(interactive=str(mode).strip().lower() == "sway")

def _parse_optional_float(raw: str | None, label: str) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() == "none":
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be a float or blank.") from exc

def _parse_optional_int(raw: str | None, label: str) -> int | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() == "none":
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be an int or blank.") from exc

def _parse_optional_str(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() in {"none", "null", "off", "disable", "disabled", "base"}:
        return None
    return text

def _format_timings(stage_timings: list[tuple[str, float]], total_to_decode: float) -> str:
    lines = [
        "[timing] ---- request ----",
        *[f"[timing] {name}: {sec * 1000.0:.1f} ms" for name, sec in stage_timings],
        f"[timing] total_to_decode: {total_to_decode:.3f} s",
    ]
    return "\n".join(lines)

def _resolve_checkpoint_path(raw_checkpoint: str) -> str:
    checkpoint = str(raw_checkpoint).strip()
    if checkpoint == "":
        raise ValueError("checkpoint is required.")
    if is_speaker_inversion_safetensors_path(checkpoint):
        raise ValueError("Speaker embedding files cannot be used as model checkpoints.")

    path = Path(checkpoint)
    if path.is_file():
        return str(path.resolve())

    print(f"[INFO] Downloading/Resolving HF Model: {checkpoint}")
    try:
        local_dir = snapshot_download(repo_id=checkpoint)
        local_path = Path(local_dir)
        for ext in ["*.pt", "*.ckpt", "*.bin", "*.safetensors"]:
            found_files = list(local_path.glob(ext))
            if found_files:
                print(f"[INFO] Auto-detected model file: {found_files[0]}")
                return str(found_files[0])
        raise FileNotFoundError(f"No model file found in repo: {checkpoint}")
    except Exception as e:
        raise FileNotFoundError(f"Failed to resolve checkpoint '{checkpoint}': {e}")

def _build_runtime_key(checkpoint: str, model_device: str, model_precision: str, codec_device: str, codec_precision: str) -> RuntimeKey:
    checkpoint_path = _resolve_checkpoint_path(checkpoint)
    return RuntimeKey(
        checkpoint=checkpoint_path, model_device=str(model_device), codec_repo=CODEC_REPO,
        model_precision=str(model_precision), codec_device=str(codec_device),
        codec_precision=str(codec_precision), compile_model=False, compile_dynamic=False,
    )

def _describe_runtime(checkpoint: str, model_device: str, model_precision: str, codec_device: str, codec_precision: str) -> str:
    runtime_key = _build_runtime_key(checkpoint, model_device, model_precision, codec_device, codec_precision)
    runtime, reloaded = get_cached_runtime(runtime_key)
    status = "loaded model into memory" if reloaded else "model already loaded; reused existing runtime"
    notes = []
    if not runtime.model_cfg.use_caption_condition: notes.append("warning: this checkpoint does not enable caption conditioning.")
    if runtime.model_cfg.use_speaker_condition: notes.append("info: this checkpoint supports speaker conditioning, but UI runs without reference audio.")
    return "\n".join([status, f"checkpoint: {runtime_key.checkpoint}", *notes])

def _generate_single_audio(runtime, text: str, caption: str, num_steps: int, seed: int | None, seconds: float | None, duration_scale: float, t_schedule_mode: str, sway_coeff: float, cfg_guidance_mode: str, cfg_scale_text: float, cfg_scale_caption: float, cfg_scale: float | None, cfg_min_t: float, cfg_max_t: float, context_kv_cache: bool, max_text_len: int | None, max_caption_len: int | None, truncation_factor: float | None, rescale_k: float | None, rescale_sigma: float | None, lora_adapter: str | None, log_fn) -> tuple[np.ndarray, int, dict]:
    actual_seconds = seconds
    if actual_seconds is None:
        actual_seconds = _estimate_duration_from_text(text)
        log_fn(f"[INFO] Auto-estimated duration: {actual_seconds:.2f}s based on text length.")

    result = runtime.synthesize(
        SamplingRequest(text=text, caption=caption or None, ref_wav=None, ref_latent=None, no_ref=True,
            ref_normalize_db=-16.0, ref_ensure_max=True, num_candidates=1, decode_mode="sequential",
            seconds=actual_seconds, duration_scale=float(duration_scale), max_ref_seconds=30.0,
            max_text_len=max_text_len, max_caption_len=max_caption_len, num_steps=int(num_steps), seed=seed,
            cfg_guidance_mode=str(cfg_guidance_mode), cfg_scale_text=float(cfg_scale_text),
            cfg_scale_caption=float(cfg_scale_caption), cfg_scale_speaker=0.0, cfg_scale=cfg_scale,
            cfg_min_t=float(cfg_min_t), cfg_max_t=float(cfg_max_t), truncation_factor=truncation_factor,
            rescale_k=rescale_k, rescale_sigma=rescale_sigma, context_kv_cache=bool(context_kv_cache),
            speaker_kv_scale=None, speaker_kv_min_t=None, speaker_kv_max_layers=None,
            t_schedule_mode=str(t_schedule_mode), sway_coeff=float(sway_coeff), trim_tail=True, lora_adapter=lora_adapter),
        log_fn=log_fn,
    )
    audio = result.audios[0].float().cpu().numpy()
    if audio.ndim == 2: audio = audio.mean(axis=0)
    audio = np.squeeze(audio)
    if audio.ndim == 0: audio = np.array([audio])
    return audio, result.sample_rate, {"seed_used": result.used_seed, "messages": result.messages}

def _concatenate_audios(audio_list: list[np.ndarray]) -> np.ndarray:
    if not audio_list: return np.array([])
    flattened = []
    for a in audio_list:
        a = np.squeeze(a)
        if a.ndim == 0: a = np.array([a])
        if a.size == 0: continue
        flattened.append(a)
    if not flattened: return np.array([])
    return np.concatenate(flattened)

def parse_dialogue_script(script: str) -> list[tuple[str, str]]:
    """يحلل النص ويستخرج المتحدث والنص تلقائياً"""
    lines = script.strip().split('\n')
    parsed = []
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # دعم النقطتين الإنجليزية والعربية
        if ':' in line:
            parts = line.split(':', 1)
        elif '：' in line:
            parts = line.split('：', 1)
        else:
            continue # تجاهل السطور التي لا تحتوي على فاصل
        
        speaker = parts[0].strip()
        text = parts[1].strip()
        if speaker and text:
            parsed.append((speaker, text))
    return parsed

def _run_dialogue_generation(
    checkpoint: str, model_device: str, model_precision: str, codec_device: str, codec_precision: str,
    dialogue_script: str, tone1: str, tone2: str,
    num_steps: int, seed_raw: str, seconds_raw: str, duration_scale: float,
    t_schedule_mode: str, sway_coeff: float, cfg_guidance_mode: str,
    cfg_scale_text: float, cfg_scale_caption: float, cfg_scale_raw: str,
    cfg_min_t: float, cfg_max_t: float, context_kv_cache: bool,
    max_text_len_raw: str, max_caption_len_raw: str, truncation_factor_raw: str,
    rescale_k_raw: str, rescale_sigma_raw: str, lora_adapter_raw: str,
) -> tuple[str | None, str, str]:
    
    def stdout_log(msg: str) -> None:
        print(msg, flush=True)

    parsed_lines = parse_dialogue_script(dialogue_script)
    if not parsed_lines:
        raise ValueError("لم يتم العثور على حوار صالح. استخدم التنسيق: اسم المتحدث: النص")

    # معرفة أسماء المتحدثين تلقائياً
    unique_speakers = []
    for spk, _ in parsed_lines:
        if spk not in unique_speakers:
            unique_speakers.append(spk)
    
    # ربط الأسماء بالنبرات
    speaker_tone_map = {}
    if len(unique_speakers) >= 1: speaker_tone_map[unique_speakers[0]] = tone1
    if len(unique_speakers) >= 2: speaker_tone_map[unique_speakers[1]] = tone2
    # إذا وجد متحدث ثالث، سيستخدم نبرة المتحدث الأول
    if len(unique_speakers) >= 3:
        for spk in unique_speakers[2:]:
            speaker_tone_map[spk] = tone1

    stdout_log(f"Detected speakers: {unique_speakers}")
    stdout_log(f"Tone mapping: {speaker_tone_map}")

    runtime_key = _build_runtime_key(checkpoint, model_device, model_precision, codec_device, codec_precision)
    runtime, reloaded = get_cached_runtime(runtime_key)
    if not runtime.model_cfg.use_caption_condition:
        raise ValueError("Loaded checkpoint does not enable caption conditioning.")

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

    for i, (speaker, text) in enumerate(parsed_lines, 1):
        caption = speaker_tone_map.get(speaker, tone1)
        stdout_log(f"Generating Line {i}/{len(parsed_lines)}: [{speaker}] '{text}' with tone '{caption}'")
        
        try:
            audio, sr, info = _generate_single_audio(
                runtime=runtime, text=text, caption=caption, num_steps=num_steps, seed=seed,
                seconds=manual_seconds, duration_scale=duration_scale, t_schedule_mode=t_schedule_mode,
                sway_coeff=sway_coeff, cfg_guidance_mode=cfg_guidance_mode, cfg_scale_text=cfg_scale_text,
                cfg_scale_caption=cfg_scale_caption, cfg_scale=cfg_scale, cfg_min_t=cfg_min_t, cfg_max_t=cfg_max_t,
                context_kv_cache=context_kv_cache, max_text_len=max_text_len, max_caption_len=max_caption_len,
                truncation_factor=truncation_factor, rescale_k=rescale_k, rescale_sigma=rescale_sigma,
                lora_adapter=lora_adapter, log_fn=stdout_log,
            )
            audios.append(audio)
            if sample_rate is None: sample_rate = sr
            logs.append(f"Line {i} seed: {info['seed_used']}")
        except Exception as e:
            stdout_log(f"Error generating line {i}: {e}")
            logs.append(f"Error on line {i}: {str(e)}")

    merged_path = None
    if audios:
        merged_audio = _concatenate_audios(audios)
        merged_path = str(out_dir / f"dialogue_{stamp}.wav")
        sf.write(merged_path, merged_audio, sample_rate)
        stdout_log(f"Saved merged dialogue: {merged_path}")

    runtime_msg = "runtime: reloaded" if reloaded else "runtime: reused"
    detail_lines = [runtime_msg, f"Total lines processed: {len(parsed_lines)}", *logs, *([f"Merged audio: {merged_path}"] if merged_path else [])]
    detail_text = "\n".join(detail_lines)
    timing_text = "Timing information per turn printed to console."

    return merged_path, detail_text, timing_text

def _clear_runtime_cache() -> str:
    clear_cached_runtime()
    return "cleared loaded model from memory"

def build_ui() -> gr.Blocks:
    default_checkpoint = _default_checkpoint()
    default_model_device = _default_model_device()
    default_codec_device = _default_codec_device()
    device_choices = list_available_runtime_devices()
    model_precision_choices = _precision_choices_for_device(default_model_device)
    codec_precision_choices = _precision_choices_for_device(default_codec_device)

    with gr.Blocks(title="arabic-TTS Dialogue (Script Format)") as demo:
        gr.Markdown("# حوار بين متحدثين - Arabic TTS Dialogue")
        gr.Markdown("أدخل الحوار بالتنسيق التالي: `اسم المتحدث: النص` وسيتولى البرنامج تمييز الأصوات تلقائياً.")

        with gr.Row():
            checkpoint = gr.Textbox(label="Model Checkpoint", value=default_checkpoint, scale=4)
            model_device = gr.Dropdown(label="Model Device", choices=device_choices, value=default_model_device, scale=1)
            model_precision = gr.Dropdown(label="Model Precision", choices=model_precision_choices, value=model_precision_choices[0], scale=1)
            codec_device = gr.Dropdown(label="Codec Device", choices=device_choices, value=default_codec_device, scale=1)
            codec_precision = gr.Dropdown(label="Codec Precision", choices=codec_precision_choices, value=codec_precision_choices[0], scale=1)

        with gr.Row():
            load_model_btn = gr.Button("Load Model")
            clear_cache_btn = gr.Button("Unload Model")
            clear_cache_msg = gr.Textbox(label="Model Status", interactive=False)

        # واجهة إدخال الحوار الجديدة
        dialogue_script = gr.Textbox(
            label="نص الحوار", 
            lines=8, 
            placeholder="أحمد: هل انتهيت من قراءة هذا الكتاب؟\nسارة: ليس بالكامل، لكنني وصلت إلى الفصل الأخير.\nأحمد: وكيف وجدته حتى الآن؟",
            value="أحمد: هل انتهيت من قراءة هذا الكتاب؟\nسارة: ليس بالكامل، لكنني وصلت إلى الفصل الأخير.\nأحمد: وكيف وجدته حتى الآن؟\nسارة: رائع جدًا، أشعر أنه يغير طريقة تفكيري."
        )

        with gr.Row():
            with gr.Column():
                tone1 = gr.Dropdown(label="نبرة المتحدث الأول (أحمد)", choices=TONE_CHOICES, value="نبرة رسمية")
            with gr.Column():
                tone2 = gr.Dropdown(label="نبرة المتحدث الثاني (سارة)", choices=TONE_CHOICES, value="صوت انثوي نبرة رسمية")

        with gr.Accordion("إعدادات التوليد", open=True):
            with gr.Row():
                num_steps = gr.Slider(label="Num Steps", minimum=1, maximum=120, value=16, step=1)
                seed_raw = gr.Textbox(label="Seed (blank=random)", value="")
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

        gr.Markdown("### المخرجات")
        out_audio_merged = gr.Audio(label="الحوار المدمج", type="filepath", interactive=False)
        out_log = gr.Textbox(label="سجل التشغيل", lines=8)
        out_timing = gr.Textbox(label="التوقيت", lines=4)

        generate_btn.click(
            _run_dialogue_generation,
            inputs=[
                checkpoint, model_device, model_precision, codec_device, codec_precision,
                dialogue_script, tone1, tone2,
                num_steps, seed_raw, seconds_raw, duration_scale, t_schedule_mode, sway_coeff,
                cfg_guidance_mode, cfg_scale_text, cfg_scale_caption, cfg_scale_raw, cfg_min_t,
                cfg_max_t, context_kv_cache, max_text_len_raw, max_caption_len_raw,
                truncation_factor_raw, rescale_k_raw, rescale_sigma_raw, lora_adapter_raw,
            ],
            outputs=[out_audio_merged, out_log, out_timing],
        )

        model_device.change(_on_model_device_change, inputs=[model_device], outputs=[model_precision])
        codec_device.change(_on_codec_device_change, inputs=[codec_device], outputs=[codec_precision])
        t_schedule_mode.change(_on_t_schedule_mode_change, inputs=[t_schedule_mode], outputs=[sway_coeff])

        load_model_btn.click(_describe_runtime, inputs=[checkpoint, model_device, model_precision, codec_device, codec_precision], outputs=[clear_cache_msg])
        clear_cache_btn.click(_clear_runtime_cache, outputs=[clear_cache_msg])

    return demo

def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio app for script-based dialogue TTS.")
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()

    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=False)

if __name__ == "__main__":
    main()
