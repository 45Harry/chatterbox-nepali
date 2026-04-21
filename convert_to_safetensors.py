#!/usr/bin/env python3
"""
Convert .pt model files to .safetensors format for better compatibility.
This allows INT8 and 4-Bit quantized models to be loaded by the standard loader.
"""
import torch
from pathlib import Path
from safetensors.torch import save_file
import shutil

def convert_model_to_safetensors(ckpt_dir):
    """Convert t3_mtl23ls_v2.pt to t3_mtl23ls_v2.safetensors"""
    ckpt_dir = Path(ckpt_dir)
    
    pt_path = ckpt_dir / "t3_mtl23ls_v2.pt"
    safetensors_path = ckpt_dir / "t3_mtl23ls_v2.safetensors"
    
    if not pt_path.exists():
        print(f"❌ No .pt file found at {pt_path}")
        return False
    
    if safetensors_path.exists():
        print(f"⚠️  .safetensors already exists at {safetensors_path}")
        response = input("Overwrite? (y/n): ").strip().lower()
        if response != 'y':
            print("Skipped.")
            return False
    
    print(f"Loading {pt_path}...")
    try:
        state_dict = torch.load(pt_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"❌ Error loading .pt file: {e}")
        return False
    
    # Check if it's a quantized model with metadata
    is_quantized = any(key.endswith(('.scale', '.zero_point')) or '_packed_params' in key 
                      for key in state_dict.keys())
    
    if is_quantized:
        print("⚠️  Warning: This appears to be a quantized model with torch.quantization metadata.")
        print("    Conversion may lose quantization information and result in larger file size.")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            print("Cancelled.")
            return False
    
    # Filter out quantization metadata and keep only actual weights
    # This converts a quantized model to its dequantized state
    filtered_state = {}
    for key, value in state_dict.items():
        # Skip quantization metadata
        if any(x in key for x in ['.scale', '.zero_point', '_packed_params']):
            continue
        # Keep actual parameter weights
        if isinstance(value, torch.Tensor):
            filtered_state[key] = value.clone()
        else:
            filtered_state[key] = value
    
    if len(filtered_state) == 0:
        print("❌ No valid weights found after filtering quantization metadata")
        return False
    
    print(f"Converted state dict has {len(filtered_state)} parameters")
    print(f"Saving to {safetensors_path}...")
    
    try:
        save_file(filtered_state, safetensors_path)
        print(f"✅ Successfully converted to {safetensors_path}")
        
        # Show file sizes
        pt_size = pt_path.stat().st_size / (1024**3)
        st_size = safetensors_path.stat().st_size / (1024**3)
        print(f"   Original (.pt):       {pt_size:.2f} GB")
        print(f"   Converted (.safetensors): {st_size:.2f} GB")
        
        return True
    except Exception as e:
        print(f"❌ Error saving .safetensors file: {e}")
        return False

if __name__ == "__main__":
    print("🔄 Converting quantized models to SafeTensors format\n")
    
    models_to_convert = ["quantized_int8", "quantized_4bit"]
    
    for model_dir in models_to_convert:
        print(f"\n{'='*60}")
        print(f"Processing: {model_dir}")
        print(f"{'='*60}")
        convert_model_to_safetensors(model_dir)
    
    print("\n" + "="*60)
    print("✅ Conversion complete!")
    print("="*60)
