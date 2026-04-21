import shutil
from pathlib import Path
import sys
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS

try:
    import bitsandbytes as bnb
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
except ImportError:
    print("Error: bitsandbytes or accelerate not found. Please install them first.")
    exit(1)

def replace_linear_with_bnb(model):
    """
    Recursively replace nn.Linear with bnb.nn.Linear4bit
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Create a 4-bit linear layer
            new_layer = bnb.nn.Linear4bit(
                module.in_features, 
                module.out_features, 
                bias=module.bias is not None,
                compute_dtype=torch.float16, # Better for quality
                quant_type="nf4" # NormalFloat 4 is generally better
            )
            setattr(model, name, new_layer)
        else:
            replace_linear_with_bnb(module)

def quantize_4bit():
    source_dir = REPO_ROOT / "merge" / "merged_model"
    target_dir = REPO_ROOT / "quantized_4bit"
    target_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model structure from merge/merged_model...")
    # We load the wrapper but we'll modify the t3 part
    model_wrapper = ChatterboxMultilingualTTS.from_local(source_dir, "cpu")

    print("Replacing Linear layers with 4-bit versions...")
    replace_linear_with_bnb(model_wrapper.t3)

    print(f"Saving 4-bit quantized weights to {target_dir}...")
    # 4-bit models also save as .pt usually
    torch.save(model_wrapper.t3.state_dict(), target_dir / "t3_mtl23ls_v2.pt")

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

    print("\n✅ 4-Bit (NF4) Quantization complete!")
    print(f"Location: {target_dir}")

if __name__ == "__main__":
    quantize_4bit()
