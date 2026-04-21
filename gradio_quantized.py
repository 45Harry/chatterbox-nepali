import gradio as gr
import torch
import torchaudio
import numpy as np
import random
from pathlib import Path
from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Constants
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def load_quantized_model(model_type):
    """
    Load a model based on the selected quantization type.
    """
    path_map = {
        "FP16 (GPU Optimized)": "quantized_fp16",
        "4-Bit (Ultra Low VRAM)": "quantized_4bit"
    }
    
    ckpt_dir = path_map.get(model_type, "quantized_fp16")
    print(f"🚀 Loading Nepali TTS Model ({model_type}) on {DEVICE}...")
    
    try:
        model_wrapper = ChatterboxMultilingualTTS.from_local(ckpt_dir, DEVICE)
        print(f"✅ Successfully loaded model from {ckpt_dir}")
        return model_wrapper
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def generate(model, text, ref_audio, exaggeration, temperature, top_p, rep_pen, seed):
    if model is None:
        return None, "Error: Model not loaded."
    
    if seed != 0:
        set_seed(int(seed))
        
    with torch.inference_mode():
        try:
            # Generate audio
            wav = model.generate(
                text=text,
                language_id="ne",
                audio_prompt_path=ref_audio,
                exaggeration=exaggeration,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=rep_pen
            )
            
            audio_data = wav.squeeze(0).cpu().numpy()
            return (model.sr, audio_data), None
        except Exception as e:
            err_msg = f"❌ Error during generation: {e}"
            print(err_msg)
            return None, err_msg

# Build UI
with gr.Blocks(title="Chatterbox Nepali Quantized TTS") as demo:
    gr.Markdown("# 🇳🇵 Chatterbox Nepali: Quantized Edition")
    gr.Markdown("Test optimized models (FP16 for quality, 4-Bit for ultra-low VRAM).")
    
    model_state = gr.State(None)
    
    with gr.Row():
        with gr.Column(scale=1):
            model_selector = gr.Dropdown(
                choices=["FP16 (GPU Optimized)", "4-Bit (Ultra Low VRAM)"],
                value="FP16 (GPU Optimized)",
                label="Select Quantization Type"
            )
            load_btn = gr.Button("🔄 Load/Switch Model", variant="secondary")
            status_msg = gr.Markdown("*Model not loaded yet.*")
            
            gr.HTML("<hr>")
            
            input_text = gr.Textbox(
                label="Nepali Text",
                placeholder="यता लेख्नुहोस्...",
                lines=5,
                value="नमस्ते, म नेपाली एआई हुँ। मलाई तपाईंसँग कुरा गर्न पाउँदा खुसी लागेको छ।"
            )
            ref_audio = gr.Audio(label="Reference Voice (Cloning target)", type="filepath")
            
            with gr.Accordion("Advanced Settings", open=False):
                exaggeration = gr.Slider(0.0, 1.0, value=0.5, label="Exaggeration")
                temperature = gr.Slider(0.1, 1.5, value=0.8, label="Temperature")
                top_p = gr.Slider(0.0, 1.0, value=0.95, label="Top-P")
                rep_pen = gr.Slider(1.0, 2.0, value=1.1, label="Repetition Penalty")
                seed = gr.Number(value=0, label="Seed (0 for random)")
            
            generate_btn = gr.Button("Generate Nepali Speech", variant="primary")
            
        with gr.Column(scale=1):
            audio_output = gr.Audio(label="Synthesized Nepali Audio")
            error_output = gr.Markdown("")
            
            gr.Markdown("### 📊 Which one to use?")
            gr.Markdown("- **FP16**: Fast and high quality on GPU.")
            gr.Markdown("- **4-Bit**: Uses the least memory (VRAM).")

    # Load event
    load_btn.click(
        fn=load_quantized_model,
        inputs=[model_selector],
        outputs=[model_state]
    ).then(
        lambda m: f"✅ Loaded {m.t3.__class__.__name__} successfully!" if m else "❌ Failed to load.",
        inputs=[model_state],
        outputs=[status_msg]
    )
    
    # Generate event
    generate_btn.click(
        fn=generate,
        inputs=[model_state, input_text, ref_audio, exaggeration, temperature, top_p, rep_pen, seed],
        outputs=[audio_output, error_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)
