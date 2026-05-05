#!/usr/bin/env python3
"""
inference.py - توليد صوت من نص باستخدام نموذج مدرب
يتطلب: تحميل الـ checkpoint + تهيئة النموذج بـ ModelConfig
"""
import sys, torch, torchaudio, json, os, yaml
from pathlib import Path
from irodori_tts.codec import DACVAECodec
from irodori_tts.config import ModelConfig  # ← استيراد الـ dataclass

# استيراد الكلاس الرئيسي
from irodori_tts.model import TextToLatentRFDiT

def load_config_and_model(ckpt_path, config_path, device='cuda'):
    """تحميل الـ config وبناء النموذج بشكل صحيح"""
    
    # 1. قراءة ملف الـ YAML
    with open(config_path, 'r', encoding='utf-8') as f:
        raw_cfg = yaml.safe_load(f)
    
    # 2. بناء كائن ModelConfig من القاموس
    # ملاحظة: هذا يعتمد على تعريف الـ dataclass في irodori_tts/config.py
    model_cfg_dict = raw_cfg.get('model', {})
    
    # إنشاء الـ config باستخدام القيم من الملف
    # ⚠️ قد تحتاج لتعديل هذا حسب الحقول الفعلية في ModelConfig
    cfg = ModelConfig(
        latent_dim=model_cfg_dict.get('latent_dim', 32),
        latent_patch_size=model_cfg_dict.get('latent_patch_size', 1),
        text_vocab_size=model_cfg_dict.get('text_vocab_size', 64000),
        text_tokenizer_repo=model_cfg_dict.get('text_tokenizer_repo', 'sherif1313/arabic-tokenizer-tts'),
        model_dim=model_cfg_dict.get('model_dim', 1280),
        num_layers=model_cfg_dict.get('num_layers', 12),
        num_heads=model_cfg_dict.get('num_heads', 20),
        mlp_ratio=model_cfg_dict.get('mlp_ratio', 2.875),
        text_mlp_ratio=model_cfg_dict.get('text_mlp_ratio', 2.6),
        speaker_mlp_ratio=model_cfg_dict.get('speaker_mlp_ratio', 2.6),
        text_dim=model_cfg_dict.get('text_dim', 768),
        text_layers=model_cfg_dict.get('text_layers', 10),
        text_heads=model_cfg_dict.get('text_heads', 8),
        speaker_dim=model_cfg_dict.get('speaker_dim', 768),
        speaker_layers=model_cfg_dict.get('speaker_layers', 8),
        speaker_heads=model_cfg_dict.get('speaker_heads', 12),
        speaker_patch_size=model_cfg_dict.get('speaker_patch_size', 1),
        timestep_embed_dim=model_cfg_dict.get('timestep_embed_dim', 512),
        adaln_rank=model_cfg_dict.get('adaln_rank', 192),
        text_add_bos=model_cfg_dict.get('text_add_bos', False),
        caption_add_bos=model_cfg_dict.get('caption_add_bos', False),
        # أضف أي حقول أخرى مطلوبة حسب تعريف ModelConfig
    )
    
    # 3. تهيئة النموذج
    print(f"🏗️ تهيئة النموذج: TextToLatentRFDiT")
    model = TextToLatentRFDiT(cfg)
    
    # 4. تحميل الأوزان
    print(f"💾 تحميل الـ Checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get('model', ckpt.get('state_dict', ckpt))
    state = {k.replace('module.', ''): v for k, v in state.items()}
    
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys:
        print(f"⚠️ مفاتيح مفقودة: {len(result.missing_keys)}")
    if result.unexpected_keys:
        print(f"⚠️ مفاتيح غير متوقعة: {len(result.unexpected_keys)}")
    
    model.eval().to(device)
    return model, cfg, raw_cfg

def tokenize_text(text, tokenizer_repo, device, add_bos=False):
    """تحويل النص العربي إلى توكنز باستخدام AraBERT"""
    from transformers import AutoTokenizer
    
    AutoTokenizer.from_pretrained(tokenizer_repo)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    return input_ids, attention_mask

@torch.inference_mode()
def generate_latent_from_text(model, text, tokenizer_repo, device, steps=30):
    """توليد latent من نص باستخدام النموذج المدرب"""
    print(f"🎤 تحويل النص إلى latent: '{text[:60]}...'")
    
    # 1. توكنزة النص
    input_ids, attention_mask = tokenize_text(text, tokenizer_repo, device)
    
    # 2. إعداد المدخلات للنموذج
    # ملاحظة: هذا يعتمد على واجهة نموذج TextToLatentRFDiT الفعلية
    # قد تحتاج لتعديل هذا الجزء حسب تنفيذ forward() في model.py
    
    # مثال افتراضي (عدّله حسب الكود الفعلي):
    # latent = model.generate(
    #     text_input_ids=input_ids,
    #     text_mask=attention_mask,
    #     num_steps=steps,
    #     cfg_scale=1.0  # classifier-free guidance
    # )
    
    # ⚠️ مؤقتاً: نعيد latent عشوائي للتجربة فقط
    # (احذف هذا السطر واستبدله باستدعاء النموذج الفعلي عند توفر الوثائق)
    B, T = 1, 750  # batch=1, latent_steps=750
    D = model.cfg.latent_dim if hasattr(model.cfg, 'latent_dim') else 32
    latent = torch.randn(B, T, D, device=device) * 0.1
    
    return latent

def main():
    if len(sys.argv) < 4:
        print("الاستخدام: python inference.py <checkpoint.pt> <config.yaml> <text> [output.wav]")
        print("مثال: python inference.py checkpoint.pt config.yaml 'مرحباً بك' output.wav")
        sys.exit(1)
    
    ckpt_path = sys.argv[1]
    config_path = sys.argv[2]
    text = sys.argv[3]
    output_path = sys.argv[4] if len(sys.argv) > 4 else "output_inference.wav"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 بدء التوليد على الجهاز: {device}")
    
    # 1. تحميل النموذج
    model, cfg, raw_cfg = load_config_and_model(ckpt_path, config_path, device)
    
    # 2. تحميل الـ Codec
    codec = DACVAECodec.load(device=device)
    
    # 3. توليد latent من النص
    tokenizer_repo = cfg.text_tokenizer_repo if hasattr(cfg, 'text_tokenizer_repo') else raw_cfg['model']['text_tokenizer_repo']
    latent = generate_latent_from_text(model, text, tokenizer_repo, device)
    
    # 4. فك تشفير الـ latent إلى صوت
    print("🔊 فك تشفير الـ latent إلى صوت...")
    audio = codec.decode_latent(latent)
    
    # 5. حفظ الصوت
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torchaudio.save(output_path, audio[0].cpu(), codec.sample_rate)
    
    print(f"✅ تم الحفظ: {output_path}")
    print(f"📝 النص: {text}")
    print(f"🔊 المدة: {audio.shape[-1]/codec.sample_rate:.2f} ثانية")

if __name__ == "__main__":
    main()
