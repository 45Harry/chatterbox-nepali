import shutil
from pathlib import Path
import sys
import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS

def quantize_fp16():
    source_dir = REPO_ROOT / "merge" / "merged_model"
    target_dir = REPO_ROOT / "quantized_fp16"
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {source_dir}...")
    # Load to CPU first to avoid memory spikes
    model_wrapper = ChatterboxMultilingualTTS.from_local(source_dir, "cpu")

    print("Converting T3 to FP16...")
    model_wrapper.t3.half()

    print(f"Saving quantized weights to {target_dir}...")
    # Save the T3 weights as safetensors
    t3_weights = model_wrapper.t3.state_dict()
    save_file(t3_weights, target_dir / "t3_mtl23ls_v2.safetensors")

    print("Copying support files...")
    # List of files to copy from source to target
    files_to_copy = [
        "ve.pt",
        "s3gen.pt",
        "conds.pt",
        "grapheme_mtl_merged_expanded_v1.json",
        "Cangjie5_TC.json"
    ]
    
    for filename in files_to_copy:
        src_path = source_dir / filename
        if src_path.exists():
            shutil.copy(src_path, target_dir / filename)
            print(f"  Copied {filename}")
        else:
            print(f"  Warning: {filename} not found in source directory.")

    print("\n✅ FP16 Quantization complete!")
    print(f"Location: {target_dir}")

if __name__ == "__main__":
    quantize_fp16()
