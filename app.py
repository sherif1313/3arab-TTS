import io
import os

import gradio as gr
import numpy as np
import spaces
import torch
from huggingface_hub import hf_hub_download

from arabic_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_REPO = os.environ.get("MODEL_REPO", "sherif1313/3arab-TTS-500M-v2")
CODEC_REPO = "sherif1313/DACVAE-Arabic-32dim"
MAX_GRADIO_CANDIDATES = int(os.environ.get("MAX_GRADIO_CANDIDATES", "32"))
GRADIO_AUDIO_COLS_PER_ROW = 8

# Global state
_runtime: InferenceRuntime | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------


def load_models():
    global _runtime

    if _runtime is not None:
        return

    print(f"[Info] Downloading checkpoint from {MODEL_REPO}...")
    checkpoint_path = hf_hub_download(repo_id=MODEL_REPO, filename="model.safetensors")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = "bf16" if device == "cuda" else "fp32"

    key = RuntimeKey(
        checkpoint=checkpoint_path,
        model_device=device,
        codec_repo=CODEC_REPO,
        model_precision=precision,
        codec_device=device,
        codec_precision=precision,
    )

    print("[Info] Building runtime...")
    _runtime = InferenceRuntime.from_key(key)
    print("[Info] All models loaded successfully.")


# Load models at startup
load_models()


# ---------------------------------------------------------------------------
# GPU-decorated Inference
# ---------------------------------------------------------------------------


@spaces.GPU(duration=120)
def run_inference_gpu(
    text: str,
    uploaded_audio: str | None,
    num_steps: int,
    num_candidates: int,
    seed_raw: str,
    seconds_raw: str,
    duration_scale: float,
    cfg_guidance_mode: str,
    cfg_scale_text: float,
    cfg_scale_speaker: float,
    cfg_scale_raw: str,
    cfg_min_t: float,
    cfg_max_t: float,
    context_kv_cache: bool,
    truncation_factor_raw: str,
    rescale_k_raw: str,
    rescale_sigma_raw: str,
    speaker_kv_scale_raw: str,
    speaker_kv_min_t_raw: str,
    speaker_kv_max_layers_raw: str,
) -> tuple[list[tuple[int, np.ndarray]], str]:
    load_models()

    log_buffer = io.StringIO()

    def stdout_log(msg: str) -> None:
        print(msg, flush=True)
        log_buffer.write(msg + "\n")

    if not str(text).strip():
        raise gr.Error("Please enter text to synthesize.")

    cfg_scale = _parse_optional_float(cfg_scale_raw, "cfg_scale")
    truncation_factor = _parse_optional_float(truncation_factor_raw, "truncation_factor")
    rescale_k = _parse_optional_float(rescale_k_raw, "rescale_k")
    rescale_sigma = _parse_optional_float(rescale_sigma_raw, "rescale_sigma")
    speaker_kv_scale = _parse_optional_float(speaker_kv_scale_raw, "speaker_kv_scale")
    speaker_kv_min_t = _parse_optional_float(speaker_kv_min_t_raw, "speaker_kv_min_t")
    speaker_kv_max_layers = _parse_optional_int(speaker_kv_max_layers_raw, "speaker_kv_max_layers")
    seed = _parse_optional_int(seed_raw, "seed")
    manual_seconds = _parse_optional_float(seconds_raw, "seconds")
    requested_candidates = int(num_candidates)
    if requested_candidates <= 0:
        raise gr.Error("num_candidates must be >= 1.")
    if requested_candidates > MAX_GRADIO_CANDIDATES:
        raise gr.Error(f"num_candidates must be <= {MAX_GRADIO_CANDIDATES}.")

    ref_wav: str | None = None
    no_ref = True
    if uploaded_audio is not None and str(uploaded_audio).strip() != "":
        ref_wav = str(uploaded_audio)
        no_ref = False

    stdout_log(
        (
            "[Info] request: mode={} seconds={} duration_scale={} "
            "steps={} seed={} no_ref={} candidates={}"
        ).format(
            cfg_guidance_mode,
            "auto" if manual_seconds is None else manual_seconds,
            float(duration_scale),
            int(num_steps),
            "random" if seed is None else seed,
            no_ref,
            requested_candidates,
        )
    )

    result = _runtime.synthesize(
        SamplingRequest(
            text=str(text),
            ref_wav=ref_wav,
            ref_latent=None,
            no_ref=bool(no_ref),
            ref_normalize_db=-16.0,
            ref_ensure_max=True,
            num_candidates=requested_candidates,
            decode_mode="sequential",
            seconds=manual_seconds,
            duration_scale=float(duration_scale),
            max_ref_seconds=30.0,
            max_text_len=None,
            num_steps=int(num_steps),
            seed=None if seed is None else int(seed),
            cfg_guidance_mode=str(cfg_guidance_mode),
            cfg_scale_text=float(cfg_scale_text),
            cfg_scale_speaker=float(cfg_scale_speaker),
            cfg_scale=cfg_scale,
            cfg_min_t=float(cfg_min_t),
            cfg_max_t=float(cfg_max_t),
            truncation_factor=truncation_factor,
            rescale_k=rescale_k,
            rescale_sigma=rescale_sigma,
            context_kv_cache=bool(context_kv_cache),
            speaker_kv_scale=speaker_kv_scale,
            speaker_kv_min_t=speaker_kv_min_t,
            speaker_kv_max_layers=speaker_kv_max_layers,
            trim_tail=True,
        ),
        log_fn=stdout_log,
    )

    sample_rate = result.sample_rate
    audio_results: list[tuple[int, np.ndarray]] = []
    for audio in result.audios:
        waveform = audio.squeeze(0).float().numpy()
        audio_results.append((sample_rate, waveform))
    stdout_log(f"[Info] seed_used: {result.used_seed}")
    stdout_log(f"[Info] candidates: {len(result.audios)}")
    return audio_results, log_buffer.getvalue()


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def build_demo():
    MODEL_LINK = f"https://huggingface.co/{MODEL_REPO}"
    GITHUB_REPO = "https://github.com/sherif1313/3arab-TTS/"

    title = "# 3arab-TTS-500M-v2 "
    description = f"""\


ادخل النص لتوليد الصوت. يمكنك اختيارياً رفع مرجع صوتي لتقليد الصوت.

"""

    with gr.Blocks() as demo:
        gr.Markdown(title)
        gr.Markdown(description)

        text = gr.Textbox(label="Text", lines=4)
        uploaded_audio = gr.Audio(
            label="Reference Audio Upload (optional, blank = no-reference mode)",
            type="filepath",
        )

        with gr.Accordion("Sampling", open=True):
            with gr.Row():
                num_steps = gr.Slider(
                    label="Num Steps",
                    minimum=1,
                    maximum=120,
                    value=40,
                    step=1,
                )
                num_candidates = gr.Slider(
                    label="Num Candidates",
                    minimum=1,
                    maximum=MAX_GRADIO_CANDIDATES,
                    value=1,
                    step=1,
                )
                seed_raw = gr.Textbox(
                    label="Seed (blank=random)",
                    value="",
                )
                seconds_raw = gr.Textbox(
                    label="Seconds (blank=auto)",
                    value="",
                )
                duration_scale = gr.Slider(
                    label="Duration Scale",
                    minimum=0.5,
                    maximum=1.5,
                    value=1.0,
                    step=0.01,
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
                    value=3.0,
                    step=0.1,
                )
                cfg_scale_speaker = gr.Slider(
                    label="CFG Scale Speaker",
                    minimum=0.0,
                    maximum=10.0,
                    value=5.0,
                    step=0.1,
                )

        with gr.Accordion("Advanced (Optional)", open=False):
            cfg_scale_raw = gr.Textbox(label="CFG Scale Override (optional)", value="")
            with gr.Row():
                cfg_min_t = gr.Number(label="CFG Min t", value=0.5)
                cfg_max_t = gr.Number(label="CFG Max t", value=1.0)
                context_kv_cache = gr.Checkbox(label="Context KV Cache", value=True)
            with gr.Row():
                truncation_factor_raw = gr.Textbox(label="Truncation Factor (optional)", value="")
                rescale_k_raw = gr.Textbox(label="Rescale k (optional)", value="")
                rescale_sigma_raw = gr.Textbox(label="Rescale sigma (optional)", value="")
            with gr.Row():
                speaker_kv_scale_raw = gr.Textbox(label="Speaker KV Scale (optional)", value="")
                speaker_kv_min_t_raw = gr.Textbox(label="Speaker KV Min t (optional)", value="0.9")
                speaker_kv_max_layers_raw = gr.Textbox(
                    label="Speaker KV Max Layers (optional)", value=""
                )

        generate_btn = gr.Button("Generate", variant="primary")

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
                                type="numpy",
                                visible=(i == 0),
                            )
                        )
        out_log = gr.Textbox(label="Run Log", lines=6)

        def gradio_inference(
            text,
            uploaded_audio,
            num_steps,
            num_candidates,
            seed_raw,
            seconds_raw,
            duration_scale,
            cfg_guidance_mode,
            cfg_scale_text,
            cfg_scale_speaker,
            cfg_scale_raw,
            cfg_min_t,
            cfg_max_t,
            context_kv_cache,
            truncation_factor_raw,
            rescale_k_raw,
            rescale_sigma_raw,
            speaker_kv_scale_raw,
            speaker_kv_min_t_raw,
            speaker_kv_max_layers_raw,
        ):
            try:
                audio_results, log_text = run_inference_gpu(
                    text=text,
                    uploaded_audio=uploaded_audio,
                    num_steps=num_steps,
                    num_candidates=num_candidates,
                    seed_raw=seed_raw,
                    seconds_raw=seconds_raw,
                    duration_scale=duration_scale,
                    cfg_guidance_mode=cfg_guidance_mode,
                    cfg_scale_text=cfg_scale_text,
                    cfg_scale_speaker=cfg_scale_speaker,
                    cfg_scale_raw=cfg_scale_raw,
                    cfg_min_t=cfg_min_t,
                    cfg_max_t=cfg_max_t,
                    context_kv_cache=context_kv_cache,
                    truncation_factor_raw=truncation_factor_raw,
                    rescale_k_raw=rescale_k_raw,
                    rescale_sigma_raw=rescale_sigma_raw,
                    speaker_kv_scale_raw=speaker_kv_scale_raw,
                    speaker_kv_min_t_raw=speaker_kv_min_t_raw,
                    speaker_kv_max_layers_raw=speaker_kv_max_layers_raw,
                )
                audio_updates: list[object] = []
                for i in range(MAX_GRADIO_CANDIDATES):
                    if i < len(audio_results):
                        audio_updates.append(gr.update(value=audio_results[i], visible=True))
                    else:
                        audio_updates.append(gr.update(value=None, visible=False))
                return (*audio_updates, log_text)
            except Exception as e:
                raise gr.Error(str(e)) from e

        generate_btn.click(
            fn=gradio_inference,
            inputs=[
                text,
                uploaded_audio,
                num_steps,
                num_candidates,
                seed_raw,
                seconds_raw,
                duration_scale,
                cfg_guidance_mode,
                cfg_scale_text,
                cfg_scale_speaker,
                cfg_scale_raw,
                cfg_min_t,
                cfg_max_t,
                context_kv_cache,
                truncation_factor_raw,
                rescale_k_raw,
                rescale_sigma_raw,
                speaker_kv_scale_raw,
                speaker_kv_min_t_raw,
                speaker_kv_max_layers_raw,
            ],
            outputs=[*out_audios, out_log],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.queue(default_concurrency_limit=1)
    demo.launch()
