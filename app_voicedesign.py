#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import gradio as gr
from huggingface_hub import hf_hub_download

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

# ==================== الثوابت والتعديلات ====================
CODEC_REPO = "sherif1313/DACVAE-Arabic-32dim"

CAPTION_OPTIONS = [
    "نبرة طبيعية",
    "نبرة اخباريه",
    "نبرة هادئة",
    "صوت انثوي نبرة رسمية",
    "نبرة دينية",
    "نبرة رسمية هادئة",
    "صوت انثوي نبره اخباريه",
    "(بدون نبرة)",
]

# فقط اسم مستودع HuggingFace (لا حاجة لمسارات أو أسماء ملفات)
FIXED_CHECKPOINT_REPO = "sherif1313/3arab-TTS-500M-v1-VoiceDesign"

MAX_GRADIO_CANDIDATES = 32
GRADIO_AUDIO_COLS_PER_ROW = 8
# ===========================================================

def _resolve_checkpoint_path() -> str:
    """يحمّل المستودع من HuggingFace ويبحث تلقائياً عن ملف النموذج ويرجع المسار المحلي"""
    from huggingface_hub import snapshot_download
    
    # تحميل المستودع بالكامل (يتم تخزينه مؤقتاً تلقائياً ولن يتم تحميله مرة أخرى)
    local_dir = snapshot_download(repo_id=FIXED_CHECKPOINT_REPO)
    
    # البحث تلقائياً عن ملف النموذج داخل المجلد المحلي
    local_path = Path(local_dir)
    model_file = None
    
    # البحث بالأولوية عن الامتدادات المعروفة للنماذج
    for ext in ["*.pt", "*.ckpt", "*.bin", "*.safetensors"]:
        found_files = list(local_path.glob(ext))
        if found_files:
            model_file = found_files[0]
            break
            
    if not model_file:
        raise FileNotFoundError(f"No model file (.pt, .ckpt, .bin) found in repo: {FIXED_CHECKPOINT_REPO}")
        
    if is_speaker_inversion_safetensors_path(str(model_file)):
        raise ValueError("Speaker embedding files cannot be used as model checkpoints.")
        
    print(f"[INFO] Auto-detected model file: {model_file}")
    return str(model_file)


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


def _build_runtime_key(
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
) -> RuntimeKey:
    checkpoint_path = _resolve_checkpoint_path()
    return RuntimeKey(
        checkpoint=checkpoint_path,
        model_device=str(model_device),
        codec_repo=CODEC_REPO,
        model_precision=str(model_precision),
        codec_device=str(codec_device),
        codec_precision=str(codec_precision),
        compile_model=False,
        compile_dynamic=False,
    )


def _run_generation(
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
    text: str,
    caption_choice: str,
    num_steps: int,
    num_candidates: int,
    seed_raw: str,
    seconds_raw: str,
    duration_scale: float,
    t_schedule_mode: str,
    sway_coeff: float,
    cfg_guidance_mode: str,
    cfg_scale_text: float,
    cfg_scale_caption: float,
    cfg_scale_raw: str,
    cfg_min_t: float,
    cfg_max_t: float,
    context_kv_cache: bool,
    max_text_len_raw: str,
    max_caption_len_raw: str,
    truncation_factor_raw: str,
    rescale_k_raw: str,
    rescale_sigma_raw: str,
    lora_adapter_raw: str,
) -> tuple[object, ...]:
    def stdout_log(msg: str) -> None:
        print(msg, flush=True)

    runtime_key = _build_runtime_key(
        model_device=model_device,
        model_precision=model_precision,
        codec_device=codec_device,
        codec_precision=codec_precision,
    )

    text_value = str(text).strip()
    caption_value = None
    if caption_choice and caption_choice != "نبرة طبيعية":
        caption_value = str(caption_choice).strip()

    if text_value == "":
        raise ValueError("text is required.")

    requested_candidates = int(num_candidates)
    if requested_candidates <= 0:
        raise ValueError("num_candidates must be >= 1.")
    if requested_candidates > MAX_GRADIO_CANDIDATES:
        raise ValueError(f"num_candidates must be <= {MAX_GRADIO_CANDIDATES}.")

    cfg_scale = _parse_optional_float(cfg_scale_raw, "cfg_scale")
    max_text_len = _parse_optional_int(max_text_len_raw, "max_text_len")
    max_caption_len = _parse_optional_int(max_caption_len_raw, "max_caption_len")
    truncation_factor = _parse_optional_float(truncation_factor_raw, "truncation_factor")
    rescale_k = _parse_optional_float(rescale_k_raw, "rescale_k")
    rescale_sigma = _parse_optional_float(rescale_sigma_raw, "rescale_sigma")
    seed = _parse_optional_int(seed_raw, "seed")
    manual_seconds = _parse_optional_float(seconds_raw, "seconds")
    lora_adapter = _parse_optional_str(lora_adapter_raw)

    runtime, reloaded = get_cached_runtime(runtime_key)
    if not runtime.model_cfg.use_caption_condition:
        raise ValueError(
            "Loaded checkpoint does not enable caption conditioning. Use gradio_app.py for the original reference-audio model."
        )

    stdout_log(f"[gradio-caption] runtime: {'reloaded' if reloaded else 'reused'}")
    stdout_log(
        (
            "[gradio-caption] request: model_device={} model_precision={} codec_device={} codec_precision={} "
            "mode={} schedule={} sway_coeff={} seconds={} duration_scale={} steps={} seed={} candidates={}"
        ).format(
            model_device,
            model_precision,
            codec_device,
            codec_precision,
            cfg_guidance_mode,
            t_schedule_mode,
            sway_coeff,
            "auto" if manual_seconds is None else manual_seconds,
            duration_scale,
            num_steps,
            "random" if seed is None else seed,
            requested_candidates,
        )
    )
    stdout_log(
        "[gradio-caption] conditioning: text={} caption={}".format(
            "on" if text_value else "off",
            caption_value if caption_value else "off (text-only)",
        )
    )

    result = runtime.synthesize(
        SamplingRequest(
            text=text_value,
            caption=caption_value,
            ref_wav=None,
            ref_latent=None,
            no_ref=True,
            ref_normalize_db=-16.0,
            ref_ensure_max=True,
            num_candidates=requested_candidates,
            decode_mode="sequential",
            seconds=manual_seconds,
            duration_scale=float(duration_scale),
            max_ref_seconds=30.0,
            max_text_len=max_text_len,
            max_caption_len=max_caption_len,
            num_steps=int(num_steps),
            seed=None if seed is None else int(seed),
            cfg_guidance_mode=str(cfg_guidance_mode),
            cfg_scale_text=float(cfg_scale_text),
            cfg_scale_caption=float(cfg_scale_caption),
            cfg_scale_speaker=0.0,
            cfg_scale=cfg_scale,
            cfg_min_t=float(cfg_min_t),
            cfg_max_t=float(cfg_max_t),
            truncation_factor=truncation_factor,
            rescale_k=rescale_k,
            rescale_sigma=rescale_sigma,
            context_kv_cache=bool(context_kv_cache),
            speaker_kv_scale=None,
            speaker_kv_min_t=None,
            speaker_kv_max_layers=None,
            t_schedule_mode=str(t_schedule_mode),
            sway_coeff=float(sway_coeff),
            trim_tail=True,
            lora_adapter=lora_adapter,
        ),
        log_fn=stdout_log,
    )

    out_dir = Path("gradio_outputs_voicedesign")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_paths: list[str] = []
    for i, audio in enumerate(result.audios, start=1):
        out_path = save_wav(
            out_dir / f"sample_{stamp}_{i:03d}.wav",
            audio.float(),
            result.sample_rate,
        )
        out_paths.append(str(out_path))

    runtime_msg = "runtime: reloaded" if reloaded else "runtime: reused"
    detail_lines = [
        runtime_msg,
        f"seed_used: {result.used_seed}",
        f"candidates: {len(result.audios)}",
        *[f"saved[{i}]: {path}" for i, path in enumerate(out_paths, start=1)],
        *result.messages,
    ]
    if runtime.model_cfg.use_speaker_condition:
        detail_lines.append(
            "info: speaker conditioning exists in this checkpoint, but this UI forced no-reference mode."
        )
    detail_text = "\n".join(detail_lines)
    timing_text = _format_timings(result.stage_timings, result.total_to_decode)
    stdout_log(f"[gradio-caption] saved {len(out_paths)} candidates")

    audio_updates: list[object] = []
    for i in range(MAX_GRADIO_CANDIDATES):
        if i < len(out_paths):
            audio_updates.append(gr.update(value=out_paths[i], visible=True))
        else:
            audio_updates.append(gr.update(value=None, visible=False))
    return (*audio_updates, detail_text, timing_text)


def build_ui() -> gr.Blocks:
    default_model_device = _default_model_device()
    default_codec_device = _default_codec_device()
    device_choices = list_available_runtime_devices()
    model_precision_choices = _precision_choices_for_device(default_model_device)
    codec_precision_choices = _precision_choices_for_device(default_codec_device)

    with gr.Blocks(title="3arabicTTS VoiceDesign Inference") as demo:
        gr.Markdown("# 3arabic-TTS VoiceDesign Inference")
        gr.Markdown(
            "VoiceDesign model (caption-conditioned). Choose a style from the dropdown."
        )

        # صف إعدادات الجهاز والدقة (بدون checkpoint)
        with gr.Row():
            model_device = gr.Dropdown(
                label="Model Device",
                choices=device_choices,
                value=default_model_device,
                scale=1,
            )
            model_precision = gr.Dropdown(
                label="Model Precision",
                choices=model_precision_choices,
                value=model_precision_choices[0],
                scale=1,
            )
            codec_device = gr.Dropdown(
                label="Codec Device",
                choices=device_choices,
                value=default_codec_device,
                scale=1,
            )
            codec_precision = gr.Dropdown(
                label="Codec Precision",
                choices=codec_precision_choices,
                value=codec_precision_choices[0],
                scale=1,
            )

        # النص والـ caption
        with gr.Column():
            text = gr.Textbox(
                label="Text",
                lines=6,
                elem_id="arabic-voicedesign-text-input",
            )
        caption = gr.Dropdown(
            label="Caption / Style Prompt (اختر من القائمة)",
            choices=CAPTION_OPTIONS,
            value="نبرة طبيعية",
            interactive=True,
        )

        # إعدادات sampling
        with gr.Accordion("Sampling", open=True):
            with gr.Row():
                num_steps = gr.Slider(label="Num Steps", minimum=1, maximum=120, value=40, step=1)
                num_candidates = gr.Slider(
                    label="Num Candidates",
                    minimum=1,
                    maximum=MAX_GRADIO_CANDIDATES,
                    value=1,
                    step=1,
                )
                seed_raw = gr.Textbox(label="Seed (blank=random)", value="")
                seconds_raw = gr.Textbox(label="Seconds (blank=auto)", value="")
                duration_scale = gr.Slider(
                    label="Duration Scale",
                    minimum=0.5,
                    maximum=1.5,
                    value=1.0,
                    step=0.01,
                )

            with gr.Row():
                t_schedule_mode = gr.Dropdown(
                    label="Time Schedule",
                    choices=["linear", "sway"],
                    value="linear",
                )
                sway_coeff = gr.Slider(
                    label="Sway Coeff",
                    minimum=-1.0,
                    maximum=1.5,
                    value=-1.0,
                    step=0.1,
                    interactive=False,
                )

            with gr.Row():
                cfg_guidance_mode = gr.Dropdown(
                    label="CFG Guidance Mode",
                    choices=["independent", "joint", "alternating"],
                    value="independent",
                )
                cfg_scale_text = gr.Slider(
                    label="CFG Scale Text",
                    minimum=0.0,
                    maximum=10.0,
                    value=2.0,
                    step=0.1,
                )
                cfg_scale_caption = gr.Slider(
                    label="CFG Scale Caption",
                    minimum=0.0,
                    maximum=10.0,
                    value=4.0,
                    step=0.1,
                )

        with gr.Accordion("Advanced (Optional)", open=False):
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

        generate_btn = gr.Button("Generate", variant="primary")

        # مخرجات الصوت
        out_audios: list[gr.Audio] = []
        num_rows = (
            MAX_GRADIO_CANDIDATES + GRADIO_AUDIO_COLS_PER_ROW - 1
        ) // GRADIO_AUDIO_COLS_PER_ROW
        with gr.Column():
            for row_idx in range(num_rows):
                with gr.Row():
                    for col_idx in range(GRADIO_AUDIO_COLS_PER_ROW):
                        i = row_idx * GRADIO_AUDIO_COLS_PER_ROW + col_idx
                        if i >= MAX_GRADIO_CANDIDATES:
                            break
                        out_audios.append(
                            gr.Audio(
                                label=f"Generated Audio {i + 1}",
                                type="filepath",
                                interactive=False,
                                visible=(i == 0),
                                min_width=160,
                            )
                        )
        out_log = gr.Textbox(label="Run Log", lines=8)
        out_timing = gr.Textbox(label="Timing", lines=8)

        # ربط زر التوليد
        generate_btn.click(
            _run_generation,
            inputs=[
                model_device,
                model_precision,
                codec_device,
                codec_precision,
                text,
                caption,
                num_steps,
                num_candidates,
                seed_raw,
                seconds_raw,
                duration_scale,
                t_schedule_mode,
                sway_coeff,
                cfg_guidance_mode,
                cfg_scale_text,
                cfg_scale_caption,
                cfg_scale_raw,
                cfg_min_t,
                cfg_max_t,
                context_kv_cache,
                max_text_len_raw,
                max_caption_len_raw,
                truncation_factor_raw,
                rescale_k_raw,
                rescale_sigma_raw,
                lora_adapter_raw,
            ],
            outputs=[*out_audios, out_log, out_timing],
        )

        # تحديث القوائم عند تغيير الجهاز
        model_device.change(
            _on_model_device_change, inputs=[model_device], outputs=[model_precision]
        )
        codec_device.change(
            _on_codec_device_change, inputs=[codec_device], outputs=[codec_precision]
        )
        t_schedule_mode.change(
            _on_t_schedule_mode_change, inputs=[t_schedule_mode], outputs=[sway_coeff]
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gradio app for caption-conditioned arabic-TTS checkpoints."
    )
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=bool(args.share),
        debug=bool(args.debug),
    )


if __name__ == "__main__":
    main()
