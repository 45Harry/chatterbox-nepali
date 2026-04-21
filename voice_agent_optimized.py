"""
Optimized Voice Agent with GPU acceleration for low-latency inference (<300ms)
Uses streaming chunks and GPU acceleration best practices.
"""
import gradio as gr
import torch
import numpy as np
from pathlib import Path
from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS
import time
from collections import deque

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "quantized_fp16"  # Back to FP16 - base model taking too long to load
CHUNK_DURATION_MS = 256  # Process audio in chunks for real-time feel
WARMUP_ITERATIONS = 1  # Reduce warmup to speed up startup


def resolve_model_path(preferred_path: str) -> Path:
    """Pick a usable model directory, preferring quantized checkpoints."""
    candidates = [Path(preferred_path), Path("merge/merged_model"), Path("base_model")]
    for candidate in candidates:
        if (candidate / "ve.pt").exists() and (
            (candidate / "t3_mtl23ls_v2.safetensors").exists() or
            (candidate / "t3_mtl23ls_v2.pt").exists()
        ):
            return candidate

    raise FileNotFoundError(
        "No valid model directory found. Expected files like 've.pt' and "
        "'t3_mtl23ls_v2.safetensors' in one of: quantized_fp16/, merge/merged_model/, base_model/. "
        "Generate FP16 model with: python quantized_model/quantize_fp16.py"
    )

class VoiceAgentOptimized:
    def __init__(self, model_path, device):
        self.device = device
        self.model = None
        self.latency_history = deque(maxlen=100)
        self.default_ref_audio = self._resolve_default_ref_audio()
        self.load_model(model_path)

    def _resolve_default_ref_audio(self):
        """Best-effort default reference voice for runs without conds.pt."""
        candidates = [
            Path("samples/ref.wav"),
            Path("samples/achyut_ref_10s.wav"),
            Path("samples/ref2.wav"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return None
        
    def load_model(self, model_path):
        """Load model with GPU optimization"""
        print(f"🚀 Loading model on {self.device}...")
        resolved_model_path = resolve_model_path(model_path)
        print(f"📁 Using model directory: {resolved_model_path}")

        # Load model
        self.model = ChatterboxMultilingualTTS.from_local(resolved_model_path, self.device)
        
        # Set to eval mode and disable gradients (critical for inference speed)
        self.model.t3.eval()
        self.model.s3gen.eval()
        
        # Warm up GPU kernels with dummy forward passes
        print("⚡ Warming up GPU kernels...")
        self._warmup()
        
        print("✅ Model ready for inference")
        
    def _warmup(self):
        """Warm up GPU kernels for faster first inference"""
        dummy_text = "नमस्ते"
        with torch.inference_mode():
            for _ in range(WARMUP_ITERATIONS):
                try:
                    _ = self.model.generate(
                        text=dummy_text,
                        language_id="ne",
                        temperature=0.8,
                        top_p=0.95
                    )
                except:
                    pass
    
    def generate_speech(self, text, ref_audio=None, exaggeration=0.5, 
                       temperature=0.8, top_p=0.95, rep_pen=1.1):
        """Generate speech with latency tracking"""
        start = time.time()
        effective_ref_audio = ref_audio
        if not effective_ref_audio and self.model.conds is None:
            effective_ref_audio = self.default_ref_audio
            if not effective_ref_audio:
                return None, (
                    "Please upload reference audio (voice cloning input). "
                    "No built-in conds.pt or default sample reference was found."
                )
        
        with torch.inference_mode():
            try:
                # Prevent over-generation: bound token budget to text length.
                max_new_tokens = min(280, max(90, int(len(text.strip()) * 2.2)))
                wav = self.model.generate(
                    text=text,
                    language_id="ne",
                    audio_prompt_path=effective_ref_audio,
                    exaggeration=exaggeration,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=max(1.2, rep_pen),
                    cfg_weight=0.5,  # Better text adherence and cleaner stopping than cfg_weight=0
                    max_new_tokens=max_new_tokens,
                )
                
                latency_ms = (time.time() - start) * 1000
                self.latency_history.append(latency_ms)
                
                audio_data = wav.squeeze(0).cpu().numpy()
                return (self.model.sr, audio_data), latency_ms
                
            except Exception as e:
                return None, f"Error: {str(e)[:100]}"
    
    def get_stats(self):
        """Get latency statistics"""
        if not self.latency_history:
            return {
                "status": "No inference history",
                "latest": None,
                "average": None,
                "min": None,
                "max": None,
                "samples": 0,
            }
        
        history = list(self.latency_history)
        return {
            "latest": f"{history[-1]:.1f}ms",
            "average": f"{np.mean(history):.1f}ms",
            "min": f"{np.min(history):.1f}ms",
            "max": f"{np.max(history):.1f}ms",
            "samples": len(history)
        }

# Initialize agent
print(f"Initializing Voice Agent (Device: {DEVICE})")
agent = VoiceAgentOptimized(MODEL_PATH, DEVICE)

# Build UI
with gr.Blocks(title="🎤 Nepali Voice Agent - Optimized") as demo:
    gr.Markdown("# 🎤 Nepali Voice Agent - GPU Optimized")
    gr.Markdown("Low-latency TTS for real-time voice interactions using FP16 GPU acceleration.")
    
    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label="💬 Nepali Text Input",
                placeholder="यता लेख्नुहोस्...",
                lines=4,
                value="नमस्ते, मैले तपाईंको सहायता गर्न सक्छु।"
            )
            ref_audio = gr.Audio(
                label="🎙️ Reference Voice (Optional - for voice cloning)",
                type="filepath"
            )
            
            gr.Markdown("### ⚙️ Advanced Settings")
            with gr.Row():
                exaggeration = gr.Slider(0.0, 1.0, value=0.5, label="Exaggeration")
                temperature = gr.Slider(0.1, 1.5, value=0.6, label="Temperature")
            with gr.Row():
                top_p = gr.Slider(0.0, 1.0, value=0.85, label="Top-P")
                rep_pen = gr.Slider(1.0, 2.0, value=1.25, label="Rep. Penalty")
            
            generate_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            audio_output = gr.Audio(label="🔊 Generated Speech", interactive=False)
            latency_output = gr.Textbox(label="⏱️ Latency", interactive=False)
            
            gr.Markdown("### 📊 Performance Stats")
            stats_output = gr.JSON(label="Inference Statistics")

    def on_generate(text, ref_audio, exaggeration, temperature, top_p, rep_pen):
        """Generate with latency tracking"""
        if not text.strip():
            return None, "Error: Empty text", {}
        
        audio_result, latency = agent.generate_speech(
            text, ref_audio, exaggeration, temperature, top_p, rep_pen
        )
        
        if audio_result is None:
            return None, f"❌ {latency}", agent.get_stats()
        
        latency_str = f"✅ **{latency:.1f}ms**" if latency < 300 else f"⚠️ {latency:.1f}ms"
        
        return audio_result, latency_str, agent.get_stats()
    
    generate_btn.click(
        fn=on_generate,
        inputs=[input_text, ref_audio, exaggeration, temperature, top_p, rep_pen],
        outputs=[audio_output, latency_output, stats_output]
    )

if __name__ == "__main__":
    print("\n" + "="*70)
    print("✅ VOICE AGENT READY")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH} (FP16 for GPU optimization)")
    print(f"Target Latency: <300ms")
    print("="*70 + "\n")
    
    demo.launch(share=False)
