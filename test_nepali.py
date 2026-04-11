#!/usr/bin/env python3
"""
Nepali TTS inference script using a custom checkpoint.

Usage:
    python3 test_nepali.py \
      --checkpoint results/t3_nepali_epoch_45.pt \
      --ref_audio samples/ref.wav \
      --text "नमस्ते, म टेलभोक्सको आवाज बोल्दै छु। टेलभोक्सले तपाईंको व्यवसायको लागि स्मार्ट, प्राविधिक र भरपर्दो समाधान ल्याउँछ।" \
      --output output_audio/test_output13.wav
"""
import argparse
import os
import sys
import torch
import torchaudio

# Ensure PYTHONPATH includes src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

def generate(args):
    device = torch.device(args.device)

    print(f"📦 Loading model from {args.checkpoint}...")

    # Load the full pretrained model wrapper first
    if args.ckpt_dir:
        model_wrapper = ChatterboxMultilingualTTS.from_local(args.ckpt_dir, device)
    else:
        model_wrapper = ChatterboxMultilingualTTS.from_pretrained(device)

    # Load our custom checkpoint
    print(f"🔄 Applying custom weights from {args.checkpoint}...")
    if args.checkpoint.endswith(".safetensors"):
        from safetensors.torch import load_file as load_safetensors
        state_dict = load_safetensors(args.checkpoint, device="cpu")
    else:
        state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    # Clean keys if needed
    cleaned = {k.replace("patched_model.", "").replace("model.", ""): v for k, v in state_dict.items()}
    model_wrapper.t3.load_state_dict(cleaned, strict=False)
    model_wrapper.t3.to(device)
    model_wrapper.t3.eval()

    print(f"🎤 Generating speech...")
    print(f"   Text: {args.text}")
    print(f"   Reference: {args.ref_audio}")

    with torch.no_grad():
        wav = model_wrapper.generate(
            args.text,
            language_id="ne",
            audio_prompt_path=args.ref_audio,
            exaggeration=args.exaggeration,
            temperature=args.temperature,
        )

    torchaudio.save(args.output, wav, model_wrapper.sr)
    print(f"✅ Saved to {args.output}")


if __name__ == "__main__":
    def get_default_device():
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    parser = argparse.ArgumentParser(description="Nepali TTS Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to t3_nepali_epoch_X.pt")
    parser.add_argument("--ckpt_dir", type=str, help="Path to base pretrained model dir")
    parser.add_argument("--ref_audio", type=str, required=False, help="Reference audio for voice cloning")
    parser.add_argument("--text", type=str, required=True, help="Nepali text to synthesize")
    parser.add_argument("--output", type=str, default="output.wav", help="Output WAV path")
    parser.add_argument("--device", type=str, default=get_default_device())
    parser.add_argument("--exaggeration", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.8)

    args = parser.parse_args()
    generate(args)


 