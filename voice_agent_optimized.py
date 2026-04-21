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
import re
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
        self.latency_history = deque(maxlen=100)  # total generation latency
        self.ttf_history = deque(maxlen=100)  # time-to-first-chunk
        self.default_ref_audio = self._resolve_default_ref_audio()
        self.active_ref_audio = None
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

        # Precompute conditionals once to avoid re-encoding reference audio every request.
        if self.model.conds is None and self.default_ref_audio:
            print(f"🎙️ Precomputing default voice conditionals from {self.default_ref_audio} ...")
            self.model.prepare_conditionals(self.default_ref_audio, exaggeration=0.5)
            self.active_ref_audio = self.default_ref_audio
        
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

    def _split_text_chunks(self, text: str, max_chunk_chars: int = 90):
        """Split text into chunks while avoiding boundary repetition."""
        cleaned = " ".join(text.strip().split())
        if not cleaned:
            return []

        # Split by sentence-ending punctuation while keeping punctuation.
        raw_parts = re.split(r"([।.!?])", cleaned)
        sentences = []
        for i in range(0, len(raw_parts), 2):
            core = raw_parts[i].strip()
            if not core:
                continue
            end = raw_parts[i + 1] if i + 1 < len(raw_parts) else ""
            sentences.append((core + end).strip())

        if not sentences:
            sentences = [cleaned]

        chunks = []
        for sentence in sentences:
            if len(sentence) <= max_chunk_chars:
                chunks.append(sentence)
                continue

            # Split long sentence by words (no overlap) to avoid semantic duplication.
            words = sentence.split()
            current_words = []
            current_len = 0
            for w in words:
                add_len = len(w) if current_len == 0 else len(w) + 1
                if current_len + add_len <= max_chunk_chars:
                    current_words.append(w)
                    current_len += add_len
                else:
                    if current_words:
                        chunks.append(" ".join(current_words))
                    current_words = [w]
                    current_len = len(w)
            if current_words:
                chunks.append(" ".join(current_words))

        chunks = [c for c in chunks if c]
        return self._dedupe_chunk_boundaries(chunks)

    def _dedupe_chunk_boundaries(self, chunks):
        """Remove repeated leading words in chunk N that duplicate tail of chunk N-1."""
        if len(chunks) < 2:
            return chunks

        deduped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_words = deduped[-1].split()
            cur_words = chunks[i].split()

            max_overlap = min(6, len(prev_words), len(cur_words))
            overlap = 0
            for k in range(max_overlap, 0, -1):
                if prev_words[-k:] == cur_words[:k]:
                    overlap = k
                    break

            if overlap > 0:
                cur_words = cur_words[overlap:]

            cleaned = " ".join(cur_words).strip()
            if cleaned:
                deduped.append(cleaned)

        return deduped

    def _force_small_first_chunk(self, chunks, max_words=6):
        """Create a tiny first chunk for faster TTF."""
        if not chunks:
            return chunks
        first_words = chunks[0].split()
        if len(first_words) <= max_words:
            return chunks
        head = " ".join(first_words[:max_words]).strip()
        tail = " ".join(first_words[max_words:]).strip()
        rem = [tail] + chunks[1:] if tail else chunks[1:]
        return [head] + rem

    def _ensure_conditionals(self, ref_audio, exaggeration):
        if ref_audio:
            if self.active_ref_audio != ref_audio or self.model.conds is None:
                self.model.prepare_conditionals(ref_audio, exaggeration=exaggeration)
                self.active_ref_audio = ref_audio
        elif self.model.conds is None:
            if not self.default_ref_audio:
                return (
                    "Please upload reference audio (voice cloning input). "
                    "No built-in conds.pt or default sample reference was found."
                )
            self.model.prepare_conditionals(self.default_ref_audio, exaggeration=exaggeration)
            self.active_ref_audio = self.default_ref_audio
        return None

    def generate_speech_stream(self, text, ref_audio=None, exaggeration=0.5, 
                              temperature=0.8, top_p=0.95, rep_pen=1.1, ultra_low_latency=False):
        """Yield partial audio so playback can start before full synthesis finishes."""
        start = time.time()
        cond_error = self._ensure_conditionals(ref_audio, exaggeration)
        if cond_error:
            yield None, f"❌ {cond_error}", self.get_stats()
            return

        with torch.inference_mode():
            try:
                chunks = self._split_text_chunks(text, max_chunk_chars=60)
                if ultra_low_latency:
                    chunks = self._force_small_first_chunk(chunks, max_words=6)
                if not chunks:
                    yield None, "❌ Error: Empty text", self.get_stats()
                    return

                audio_chunks = []
                first_chunk_latency_ms = None
                for idx, chunk in enumerate(chunks):
                    if ultra_low_latency and idx == 0:
                        chunk_tokens = min(48, max(26, int(len(chunk) * 1.4)))
                        chunk_temp = min(temperature, 0.45)
                        chunk_top_p = min(top_p, 0.72)
                    else:
                        chunk_tokens = min(180, max(60, int(len(chunk) * 2.6)))
                        chunk_temp = temperature
                        chunk_top_p = top_p
                    wav = self.model.generate(
                        text=chunk,
                        language_id="ne",
                        audio_prompt_path=None,
                        exaggeration=exaggeration,
                        temperature=chunk_temp,
                        top_p=chunk_top_p,
                        repetition_penalty=max(1.2, rep_pen),
                        cfg_weight=0.5,
                        max_new_tokens=chunk_tokens,
                    )
                    if idx == 0:
                        first_chunk_latency_ms = (time.time() - start) * 1000
                    audio_chunks.append(wav.squeeze(0).cpu().numpy())
                    partial_audio = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
                    elapsed_ms = (time.time() - start) * 1000
                    yield (
                        (self.model.sr, partial_audio),
                        f"⏳ TTF {first_chunk_latency_ms:.1f}ms | Elapsed {elapsed_ms:.1f}ms | Chunk {idx + 1}/{len(chunks)}"
                        + (" | Ultra" if ultra_low_latency else ""),
                        self.get_stats(),
                    )

                audio_data = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
                total_latency_ms = (time.time() - start) * 1000
                self.latency_history.append(total_latency_ms)
                if first_chunk_latency_ms is not None:
                    self.ttf_history.append(first_chunk_latency_ms)
                yield (
                    (self.model.sr, audio_data),
                    {"ttf_ms": first_chunk_latency_ms, "total_ms": total_latency_ms, "chunks": len(chunks)},
                )
                
            except Exception as e:
                yield None, f"❌ Error: {str(e)[:160]}", self.get_stats()
    
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
        ttf = list(self.ttf_history)
        return {
            "latest_total": f"{history[-1]:.1f}ms",
            "average_total": f"{np.mean(history):.1f}ms",
            "min_total": f"{np.min(history):.1f}ms",
            "max_total": f"{np.max(history):.1f}ms",
            "latest_ttf": f"{ttf[-1]:.1f}ms" if ttf else None,
            "average_ttf": f"{np.mean(ttf):.1f}ms" if ttf else None,
            "samples": len(history),
        }

# Initialize agent
print(f"Initializing Voice Agent (Device: {DEVICE})")
if DEVICE != "cuda":
    print("⚠️ CUDA not detected. Sub-300ms target is typically not achievable on CPU.")
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
            ultra_mode = gr.Checkbox(value=True, label="Ultra Low Latency (faster first chunk)")
            
            generate_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            audio_output = gr.Audio(label="🔊 Generated Speech", interactive=False)
            latency_output = gr.Textbox(label="⏱️ Latency", interactive=False)
            
            gr.Markdown("### 📊 Performance Stats")
            stats_output = gr.JSON(label="Inference Statistics")

    def on_generate(text, ref_audio, exaggeration, temperature, top_p, rep_pen, ultra_mode):
        """Stream partial audio chunks with latency tracking"""
        if not text.strip():
            yield None, "❌ Error: Empty text", {}
            return

        for event in agent.generate_speech_stream(
            text, ref_audio, exaggeration, temperature, top_p, rep_pen, ultra_mode
        ):
            if len(event) == 3:
                yield event
            else:
                audio_result, latency = event
                ttf_ms = latency["ttf_ms"]
                total_ms = latency["total_ms"]
                chunks = latency["chunks"]
                if ttf_ms is not None and ttf_ms < 300:
                    latency_str = f"✅ TTF {ttf_ms:.1f}ms | Total {total_ms:.1f}ms | Chunks {chunks}"
                else:
                    latency_str = f"⚠️ TTF {ttf_ms:.1f}ms | Total {total_ms:.1f}ms | Chunks {chunks}"
                yield audio_result, latency_str, agent.get_stats()
    
    generate_btn.click(
        fn=on_generate,
        inputs=[input_text, ref_audio, exaggeration, temperature, top_p, rep_pen, ultra_mode],
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
    
    demo.launch(share=True)
