
#!uv run python app.py
import gradio as gr
from pathlib import Path
from datetime import datetime
from huggingface_hub import hf_hub_download  
from arabic_tts.inference_runtime import RuntimeKey, SamplingRequest, get_cached_runtime, save_wav

CHECKPOINT_ID = "sherif1313/3arab-TTS-500M-v1" 
CODEC_REPO = "sherif1313/DACVAE-Arabic-32dim"

def get_local_checkpoint(repo_id: str) -> str:

    for ext in [".pt", ".safetensors", ".bin", ".ckpt"]:
        try:
            return hf_hub_download(repo_id=repo_id, filename=f"checkpoint{ext}", cache_dir=".cache")
        except: continue

    return hf_hub_download(repo_id=repo_id, filename="model.safetensors", cache_dir=".cache")

CHECKPOINT_PATH = get_local_checkpoint(CHECKPOINT_ID) 

def estimate_seconds(text: str) -> float:
    return max(3.0, min(20.0, len(text.strip()) / 10 * 1.3))

def generate(m_dev, m_prec, c_dev, c_prec, text, ref, steps, cands):
    if not text: return [], "⚠️ أدخل النص أولاً"
    
    key = RuntimeKey(
        checkpoint=CHECKPOINT_PATH,  
        model_device=m_dev, model_precision=m_prec,
        codec_repo=CODEC_REPO, codec_device=c_dev, codec_precision=c_prec,
        compile_model=False, compile_dynamic=False
    )
    
    runtime, _ = get_cached_runtime(key)
    secs = estimate_seconds(text)
    
    res = runtime.synthesize(SamplingRequest(
        text=text, ref_wav=ref, no_ref=ref is None, seconds=secs,
        num_steps=int(steps), num_candidates=int(cands), decode_mode="sequential"
    ))
    
    out = Path("out"); out.mkdir(exist_ok=True)
    paths = []
    for i, audio in enumerate(res.audios):
        p = out / f"gen_{datetime.now().strftime('%H%M%S')}_{i}.wav"
        save_wav(p, audio, res.sample_rate)
        paths.append(str(p))
        
    return paths, f"✅ تم التوليد | مدة: {secs:.1f}ث\n" + "\n".join(res.messages)

with gr.Blocks(title="TTS سريع") as app:
    gr.Markdown(f"### 🎙️ مولد صوت عربي | النموذج: `{CHECKPOINT_ID}`")
    with gr.Row():
        with gr.Column(scale=1):
            txt = gr.Textbox(label="النص", placeholder="اكتب النص هنا...")
            ref = gr.Audio(label="ملف مرجعي", type="filepath")
            steps = gr.Slider(10, 80, value=40, step=1, label="خطوات التوليد")
            cands = gr.Slider(1, 4, value=1, step=1, label="عدد المرشحين")
            d1 = gr.Dropdown(["cuda","cpu"], value="cuda", label="جهاز النموذج")
            d2 = gr.Dropdown(["bf16","fp32"], value="fp32", label="دقة النموذج")
            d3 = gr.Dropdown(["cuda","cpu"], value="cuda", label="جهاز الكوديك")
            d4 = gr.Dropdown(["bf16","fp32"], value="fp32", label="دقة الكوديك")
            btn = gr.Button("🔊 توليد", variant="primary")
        with gr.Column(scale=1):
            out_audio = gr.Files(label="الملفات المولدة")
            log = gr.Textbox(label="السجل", lines=4)
    btn.click(generate, inputs=[d1,d2,d3,d4,txt,ref,steps,cands], outputs=[out_audio, log])

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
