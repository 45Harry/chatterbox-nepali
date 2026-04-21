import os
import shutil
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS

def main():
    results_dir = Path("results")
    safetensor_files = list(results_dir.glob("*.safetensors"))
    if not safetensor_files:
        print("Error: No .safetensors found in results/")
        return
    
    # Find latest by modification time
    latest_safetensors = max(safetensor_files, key=os.path.getmtime)
    print(f"Found latest fine-tuned checkpoint: {latest_safetensors}")

    merge_dir = Path("merge/merged_model")
    if merge_dir.exists():
        shutil.rmtree(merge_dir)
        
    print(f"Copying base model to {merge_dir}...")
    shutil.copytree("base_model", merge_dir)

    print("Loading base model into memory (CPU)...")
    device = "cpu"
    model_wrapper = ChatterboxMultilingualTTS.from_local("base_model", device)

    print("Merging fine-tuned weights over base model...")
    resume_state = load_file(latest_safetensors, device=device)
    
    # Clean keys dynamically as done in api.py
    cleaned_state = {k.replace("patched_model.", "").replace("model.", ""): v for k, v in resume_state.items()}
    model_wrapper.t3.load_state_dict(cleaned_state, strict=False)

    output_safetensor = merge_dir / "t3_mtl23ls_v2.safetensors"
    print(f"Saving newly merged safetensors to {output_safetensor}...")
    save_file(model_wrapper.t3.state_dict(), output_safetensor)
    
    print("Done! You can now use 'merge/merged_model' identically to 'base_model'.")

if __name__ == "__main__":
    main()
