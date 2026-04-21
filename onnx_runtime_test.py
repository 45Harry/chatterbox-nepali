#!/usr/bin/env python3
"""
Export the T3 transformer and speech-head to ONNX and validate them with ONNX Runtime.
"""
import os
import time
from pathlib import Path

import torch
import onnxruntime as ort
from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS

MODEL_DIR = Path("quantized_fp16")
ONNX_DIR = Path("onnx_models")
ONNX_DIR.mkdir(exist_ok=True)


def export_t3_transformer(model: ChatterboxMultilingualTTS, output_path: Path) -> bool:
    print(f"Exporting T3 transformer to ONNX: {output_path}")
    model.t3.tfmr.eval()

    hidden_size = model.t3.tfmr.config.hidden_size
    seq_len = 16
    dummy_inputs = torch.randn(1, seq_len, hidden_size, dtype=torch.float32)

    try:
        torch.onnx.export(
            model.t3.tfmr,
            (dummy_inputs,),
            str(output_path),
            opset_version=18,
            input_names=["inputs_embeds"],
            output_names=["last_hidden_state"],
            do_constant_folding=True,
            verbose=False,
        )
        print("  ✅ Exported T3 transformer")
        return True
    except Exception as exc:
        print("  ⚠️  Failed to export T3 transformer to ONNX:")
        print(f"    {type(exc).__name__}: {exc}")
        return False


def export_speech_head(model: ChatterboxMultilingualTTS, output_path: Path):
    print(f"Exporting speech head to ONNX: {output_path}")
    model.t3.speech_head.eval()

    in_features = model.t3.speech_head.in_features
    seq_len = 16
    dummy_hidden = torch.randn(1, seq_len, in_features, dtype=torch.float32)

    torch.onnx.export(
        model.t3.speech_head,
        dummy_hidden,
        str(output_path),
        opset_version=18,
        input_names=["hidden_states"],
        output_names=["logits"],
        do_constant_folding=True,
        verbose=False,
    )
    print("  ✅ Exported speech head")


def verify_onnx_session(model: ChatterboxMultilingualTTS, model_path: Path, input_tensor: torch.Tensor):
    print(f"Loading ONNX model: {model_path}")
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]
    print(f"  inputs: {input_name}, outputs: {output_names}")

    input_data = input_tensor.cpu().numpy()
    outputs = sess.run(None, {input_name: input_data})
    print(f"  ✅ ONNX runtime executed {len(outputs)} outputs")
    for idx, out in enumerate(outputs):
        print(f"    output[{idx}] shape={out.shape}, dtype={out.dtype}")
    return outputs


def main():
    print("=== ONNX Runtime Experiment ===")
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

    print(f"Loading model from {MODEL_DIR} on CPU...")
    model = ChatterboxMultilingualTTS.from_local(str(MODEL_DIR), "cpu")
    print("  ✅ Model loaded")

    t3_path = ONNX_DIR / "t3_tfmr.onnx"
    speech_head_path = ONNX_DIR / "t3_speech_head.onnx"

    t3_exported = False
    if not t3_path.exists():
        t3_exported = export_t3_transformer(model, t3_path)
    else:
        print(f"Skipping export; file already exists: {t3_path}")
        t3_exported = True

    if not speech_head_path.exists():
        export_speech_head(model, speech_head_path)
    else:
        print(f"Skipping export; file already exists: {speech_head_path}")

    print("\nVerifying ONNX runtime execution...\n")
    hidden_size = model.t3.tfmr.config.hidden_size
    seq_len = 16
    dummy_inputs = torch.randn(1, seq_len, hidden_size, dtype=torch.float32)

    if t3_exported:
        t3_outputs = verify_onnx_session(model, t3_path, dummy_inputs)
        if len(t3_outputs) > 0:
            hidden = torch.from_numpy(t3_outputs[0])
            _ = verify_onnx_session(model, speech_head_path, hidden)
    else:
        print("  ⚠️  Skipping T3 transformer ONNX runtime verification because export failed.")
        dummy_hidden = torch.randn(1, seq_len, model.t3.speech_head.in_features, dtype=torch.float32)
        _ = verify_onnx_session(model, speech_head_path, dummy_hidden)

    print("\n✅ ONNX Runtime test complete")
    print(f"ONNX files written to: {ONNX_DIR.absolute()}")


if __name__ == "__main__":
    main()
