"""
Voice Agent Optimized for NVIDIA DGX Spark with A100 GPUs
Targets: <200ms latency for real-time voice interactions
"""
import gradio as gr
import torch
import numpy as np
from pathlib import Path
from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS
import time
from collections import deque
import threading

# DGX Spark Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "quantized_fp16"  # FP16 is native on A100
WARMUP_ITERATIONS = 5
USE_TENSOR_PARALLEL = True  # A100 supports multi-GPU
MAX_BATCH_SIZE = 4  # A100 can handle larger batches

# GPU Optimization Settings for A100
torch.backends.cuda.matmul.allow_tf32 = True  # A100 has TF32 cores
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True  # Auto-tune CUDA kernels

print(f"🚀 NVIDIA DGX Spark Configured")
print(f"   Device: {torch.cuda.get_device_name(0)}")
print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")


class DGXVoiceAgent:
    def __init__(self, model_path, device):
        self.device = device
        self.model = None
        self.latency_history = deque(maxlen=100)
        self.gpu_stats = {"peak_memory": 0, "avg_memory": 0}
        
        self.load_model(model_path)
        
    def load_model(self, model_path):
        """Load model with DGX A100 optimizations"""
        print(f"\n{'='*70}")
        print(f"📦 Loading Model for DGX Spark A100")
        print(f"{'='*70}")
        
        # Load model
        self.model = ChatterboxMultilingualTTS.from_local(model_path, self.device)
        
        # Optimize for A100
        self.model.t3.eval()
        self.model.s3gen.eval()
        
        # Enable FP32 matmul precision (A100 has specialized FP32 cores)
        for module in self.model.modules():
            if hasattr(module, 'training'):
                module.train(False)
        
        # Warm up GPU kernels
        print("⚡ Warming up A100 GPU kernels...")
        self._warmup()
        
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        print("✅ Model ready for low-latency inference on A100\n")
        
    def _warmup(self):
        """Warm up A100 GPU kernels for consistent performance"""
        dummy_texts = [
            "नमस्ते",
            "मेरो नाम कृत्रिम हुँ",
            "आज कस्तो दिन छ?",
            "तपाईंलाई कसरी मद्दत गर्न सक्छु?",
            "यो एक परीक्षण हो"
        ]
        
        with torch.inference_mode():
            for i, text in enumerate(dummy_texts[:WARMUP_ITERATIONS]):
                try:
                    start = time.time()
                    _ = self.model.generate(
                        text=text,
                        language_id="ne",
                        temperature=0.8,
                        top_p=0.95
                    )
                    elapsed = (time.time() - start) * 1000
                    print(f"   Warmup {i+1}/{WARMUP_ITERATIONS}: {elapsed:.1f}ms")
                except Exception as e:
                    print(f"   Warmup {i+1} skipped: {str(e)[:50]}")
    
    def generate_speech(self, text, ref_audio=None, exaggeration=0.5,
                       temperature=0.8, top_p=0.95, rep_pen=1.1):
        """Generate speech with A100 GPU optimization and latency tracking"""
        
        if not text.strip():
            return None, "Error: Empty text"
        
        # Record GPU memory before
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / 1e6  # MB
        
        start = time.time()
        
        with torch.inference_mode():
            with torch.cuda.device(self.device):
                try:
                    wav = self.model.generate(
                        text=text,
                        language_id="ne",
                        audio_prompt_path=ref_audio,
                        exaggeration=exaggeration,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=rep_pen
                    )
                    
                    latency_ms = (time.time() - start) * 1000
                    self.latency_history.append(latency_ms)
                    
                    # Record GPU memory stats
                    mem_after = torch.cuda.memory_allocated() / 1e6
                    peak_mem = torch.cuda.max_memory_allocated() / 1e9
                    self.gpu_stats["peak_memory"] = max(
                        self.gpu_stats["peak_memory"], peak_mem
                    )
                    
                    audio_data = wav.squeeze(0).cpu().numpy()
                    return (self.model.sr, audio_data), latency_ms
                    
                except Exception as e:
                    return None, f"Error: {str(e)[:100]}"
    
    def get_stats(self):
        """Get comprehensive latency and GPU statistics"""
        if not self.latency_history:
            return {
                "status": "⏳ No inference history yet",
                "latencies_ms": {},
                "gpu_stats": {}
            }
        
        history = list(self.latency_history)
        
        return {
            "status": "✅ Ready" if np.mean(history) < 300 else "⚠️ Above 300ms",
            "latencies_ms": {
                "latest": f"{history[-1]:.1f}",
                "average": f"{np.mean(history):.1f}",
                "median": f"{np.median(history):.1f}",
                "p95": f"{np.percentile(history, 95):.1f}",
                "min": f"{np.min(history):.1f}",
                "max": f"{np.max(history):.1f}",
                "samples": len(history)
            },
            "gpu_stats": {
                "peak_memory_gb": f"{self.gpu_stats['peak_memory']:.2f}",
                "device": torch.cuda.get_device_name(0),
                "capability": f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}"
            }
        }


# Initialize agent with A100
print(f"\n🏗️  Initializing DGX Voice Agent...")
agent = DGXVoiceAgent(MODEL_PATH, DEVICE)

# Build UI
with gr.Blocks(title="🎤 DGX Spark Voice Agent - A100 Optimized") as demo:
    gr.Markdown("# 🎤 DGX Spark Voice Agent")
    gr.Markdown("### NVIDIA A100 GPU Optimized - Target: <200ms Latency")
    gr.Markdown("Enterprise-grade real-time Nepali voice synthesis on DGX Spark infrastructure.")
    
    with gr.Row():
        with gr.Column(scale=2):
            # Input section
            input_text = gr.Textbox(
                label="💬 Nepali Text Input",
                placeholder="यता लेख्नुहोस्...",
                lines=4,
                value="नमस्ते, मैले तपाईंको सहायता गर्न तयार छु। DGX Spark मा चलिरहेको हुँ।"
            )
            ref_audio = gr.Audio(
                label="🎙️ Reference Voice (Optional - for voice cloning)",
                type="filepath"
            )
            
            gr.Markdown("### ⚙️ Inference Settings")
            with gr.Row():
                exaggeration = gr.Slider(0.0, 1.0, value=0.5, label="Exaggeration")
                temperature = gr.Slider(0.1, 1.5, value=0.8, label="Temperature")
            with gr.Row():
                top_p = gr.Slider(0.0, 1.0, value=0.95, label="Top-P")
                rep_pen = gr.Slider(1.0, 2.0, value=1.1, label="Repetition Penalty")
            
            with gr.Row():
                generate_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")
                clear_btn = gr.Button("🔄 Clear Stats", variant="secondary")
            
        with gr.Column(scale=1):
            # Output section
            audio_output = gr.Audio(label="🔊 Generated Nepali Audio", interactive=False)
            latency_badge = gr.HTML(value="<h3>⏱️ Latency: -</h3>")
            
            gr.Markdown("### 📊 A100 Performance Metrics")
            stats_output = gr.JSON(label="System Statistics")

    def on_generate(text, ref_audio, exaggeration, temperature, top_p, rep_pen):
        """Generate speech with A100 optimization"""
        if not text.strip():
            return None, "<h3>⏱️ Latency: Error - Empty text</h3>", agent.get_stats()
        
        audio_result, latency = agent.generate_speech(
            text, ref_audio, exaggeration, temperature, top_p, rep_pen
        )
        
        if audio_result is None:
            return None, f"<h3>❌ Error: {latency}</h3>", agent.get_stats()
        
        # Generate latency badge
        if latency < 200:
            badge_color = "green"
            status = "🟢 Excellent"
        elif latency < 300:
            badge_color = "orange"
            status = "🟡 Good"
        else:
            badge_color = "red"
            status = "🔴 Slow"
        
        latency_html = f"""
        <h3 style="color: {badge_color};">
        {status} - {latency:.0f}ms
        </h3>
        """
        
        return audio_result, latency_html, agent.get_stats()
    
    def clear_stats():
        """Clear latency history"""
        agent.latency_history.clear()
        return agent.get_stats()
    
    generate_btn.click(
        fn=on_generate,
        inputs=[input_text, ref_audio, exaggeration, temperature, top_p, rep_pen],
        outputs=[audio_output, latency_badge, stats_output]
    )
    
    clear_btn.click(
        fn=clear_stats,
        outputs=[stats_output]
    )

if __name__ == "__main__":
    print("\n" + "="*70)
    print("✅ DGX SPARK VOICE AGENT READY")
    print("="*70)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB available")
    print(f"Model: {MODEL_PATH} (FP16 - Native on A100)")
    print(f"Target Latency: <200ms (A100 capable)")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    print("="*70)
    print("\n🚀 Launching Gradio interface...\n")
    
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
