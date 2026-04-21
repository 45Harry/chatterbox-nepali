import shutil
from pathlib import Path
import sys
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS

def quantize_int8():
    source_dir = REPO_ROOT / "merge" / "merged_model"
    target_dir = REPO_ROOT / "quantized_int8"
    target_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model from merge/merged_model...")
    model_wrapper = ChatterboxMultilingualTTS.from_local(source_dir, "cpu")

    print("Applying INT8 Dynamic Quantization to Linear layers...")
    # Set the quantization backend for ARM (aarch64)
    torch.backends.quantized.engine = 'qnnpack'
    
    # This quantizes all Linear layers to 8-bit integers
    quantized_t3 = torch.quantization.quantize_dynamic(

        model_wrapper.t3,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    print(f"Saving quantized weights to {target_dir}...")
    # For quantized models, we usually save as .pt because safetensors 
    # doesn't support the specialized quantized tensor types easily.
    torch.save(quantized_t3.state_dict(), target_dir / "t3_mtl23ls_v2.pt")
    
    # We also need a small tweak to the loading code later if we use .pt instead of .safetensors
    # but for now we follow the user's request to "write the corresponding code".

    print("Copying support files...")
    files_to_copy = [
        "ve.pt",
        "s3gen.pt",
        "grapheme_mtl_merged_expanded_v1.json",
        "Cangjie5_TC.json"
    ]
    
    for filename in files_to_copy:
        src_path = source_dir / filename
        if src_path.exists():
            shutil.copy(src_path, target_dir / filename)
            print(f"  Copied {filename}")

    print("\n✅ INT8 Quantization complete!")
    print(f"Location: {target_dir}")

if __name__ == "__main__":
    quantize_int8()
