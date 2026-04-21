import gradio as gr
import torch
import torchaudio
import numpy as np
import random
from pathlib import Path
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Constants
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "quantized_4bit/t3_mtl23ls_v2.pt"

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def load_model():
    print(f"🚀 Loading Nepali TTS Model on {DEVICE}...")
    # Load base multilingual wrapper
    model_wrapper = ChatterboxMultilingualTTS.from_pretrained(DEVICE)
    
    # Load our fine-tuned Nepali weights
    if Path(CHECKPOINT).exists():
        print(f"📦 Loading fine-tuned weights from: {CHECKPOINT}")
        from safetensors.torch import load_file
        resume_state = load_file(CHECKPOINT, device=DEVICE)
        
        # Strip prefixes if they exist (common in our training script)
        cleaned_state = {k.replace("patched_model.", "").replace("model.", ""): v for k, v in resume_state.items()}
        model_wrapper.t3.load_state_dict(cleaned_state, strict=False)
    else:
        print(f"⚠️ Warning: {CHECKPOINT} not found. Using base multilingual weights.")
    
    model_wrapper.t3.to(DEVICE).eval()
    return model_wrapper

def generate(model, text, ref_audio, exaggeration, temperature, top_p, rep_pen, seed):
    if model is None:
        return None
    
    if seed != 0:
        set_seed(int(seed))
        
    with torch.inference_mode():
        try:
            # Generate audio using our Nepali model
            wav = model.generate(
                text=text,
                language_id="ne",
                audio_prompt_path=ref_audio,
                exaggeration=exaggeration,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=rep_pen
            )
            
            # Convert to numpy for Gradio (SampleRate, AudioData)
            audio_data = wav.squeeze(0).cpu().numpy()
            return (model.sr, audio_data)
        except Exception as e:
            print(f"❌ Error during generation: {e}")
            return None

# Build UI
with gr.Blocks(title="Chatterbox Nepali TTS") as demo:
    gr.Markdown("# 🇳🇵 Chatterbox Nepali TTS")
    gr.Markdown("Fine-tuned Nepali speech synthesis for high-quality voice cloning.")
    
    model_state = gr.State(None)
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Nepali Text",
                placeholder="यता लेख्नुहोस्...",
                lines=5,
                value="नमस्ते, म नेपाली एआई हुँ। मलाई तपाईंसँग कुरा गर्न पाउँदा खुसी लागेको छ।"
            )
            ref_audio = gr.Audio(label="Reference Voice (Cloning target)", type="filepath")
            
            with gr.Accordion("Advanced Settings", open=False):
                exaggeration = gr.Slider(0.0, 1.0, value=0.5, label="Exaggeration (Pacing/Style)")
                temperature = gr.Slider(0.1, 1.5, value=0.8, label="Temperature (Randomness)")
                top_p = gr.Slider(0.0, 1.0, value=0.95, label="Top-P Sampling")
                rep_pen = gr.Slider(1.0, 2.0, value=1.1, label="Repetition Penalty")
                seed = gr.Number(value=0, label="Seed (0 for random)")
            
            generate_btn = gr.Button("Generate Nepali Speech", variant="primary")
            
        with gr.Column():
            audio_output = gr.Audio(label="Synthesized Nepali Audio")
            gr.Markdown("### Tips for Nepali:")
            gr.Markdown("- Use short, clear sentences for best results.")
            gr.Markdown("- Ensure the reference audio is clean (no background noise).")
            gr.Markdown("- Adjust **Temperature** if the voice sounds robotic or cuts off.")

    # Initialize model on load
    demo.load(load_model, outputs=[model_state])
    
    # Click event
    generate_btn.click(
        fn=generate,
        inputs=[model_state, input_text, ref_audio, exaggeration, temperature, top_p, rep_pen, seed],
        outputs=audio_output
    )

if __name__ == "__main__":
    demo.launch(share=True)
