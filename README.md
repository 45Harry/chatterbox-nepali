# 🇳🇵 Chatterbox-TTS (Nepali Edition)

Fine-tuned text-to-speech for the Nepali language, based on the **Chatterbox-Multilingual-500M** architecture. This repository contains the custom training logic, bug fixes for the Devanagari script, and tools for high-fidelity Nepali voice cloning.

![Nepali TTS Preview](./Chatterbox-Multilingual.png)

## 🚀 Key Features
* **Full Devanagari Support**: Patched tokenizer and alignment analyzer to handle complex Nepali character clusters without cutting off.
* **Seamless Voice Cloning**: High-quality zero-shot cloning of Nepali voices using a 5-10 second reference clip.
* **Optimized for Mac**: Pre-configured for **Apple Silicon (MPS)** acceleration and memory-efficient training on M2/M3 chips.
* **Clean Inference**: Dedicated Gradio UI and test scripts for rapid experimentation.

## 📦 Model Files
To ensure repository performance, large model files are hosted on Hugging Face:
- **Repo Link**: [https://huggingface.co/officialuser/chatterbox-nepali](https://huggingface.co/officialuser/chatterbox-nepali)

| File | Purpose | Recommendation |
| :--- | :--- | :--- |
| `t3_mtl_nepali_final.safetensors` | **Production Weights** | Use for **fast, optimized inference** and production use. |
| `t3_nepali_epoch_20.pt` | **Training Checkpoint** | Use as a starting point to **train further** on your own dataset. |

*Place these files in the root folder of this repository after downloading.*

---

## 🏋️ Training Further (Fine-tuning)
You are encouraged to push the model even further! To start training from the current Nepali base:

1. **Prepare Data**: Place your `.wav` files and a `metadata.csv` (format: `file|text`) in `data/nepali/`.
2. **Resume Training**:
```bash
export PYTHONPATH=src
python3 src/chatterbox/train_nepali.py \
  --manifest data/nepali/metadata.csv \
  --device mps \
  --batch_size 4 \
  --accum_steps 4 \
  --epochs 100 \
  --save_every 5 \
  --resume_t3_weights "t3_nepali_epoch_20.pt"
```
*When your training reaches the target epoch limit, the script will automatically consolidate your weights into a single optimized file: **`t3_mtl_nepali_final.safetensors`**. Please share this file back with the community!*

---

## 🎙️ Inference & Implementation

### 🛡️ Faster Generation (Safetensors)
Using the `.safetensors` format is significantly **faster** and more secure than standard `.pt` files. Use the following code to generate audio from your final model:

```python
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from safetensors.torch import load_file

# Load optimized Multilingual Wrapper
model = ChatterboxMultilingualTTS.from_pretrained(device="mps")

# Patch in your local Nepali weights
weights = load_file("t3_mtl_nepali_final.safetensors", device="mps")
cleaned_weights = {k.replace("patched_model.", "").replace("model.", ""): v for k, v in weights.items()}
model.t3.load_state_dict(cleaned_weights, strict=False)

# Synthesize Nepali
text = "नमस्ते, म नेपाली एआई एजेन्ट हुँ। म तपाईंसँग कुरा गर्न तयार छु।"
wav = model.generate(text, language_id="ne", audio_prompt_path="reference.wav")

ta.save("nepali_output.wav", wav, model.sr)
```

### 🏮 Web UI (Gradio)
Launch a graphical interface to test voices instantly:
```bash
# Automatically loads t3_mtl_nepali_final.safetensors if present
python3 gradio_nepali.py
```

## 🛠️ Critical Bug Fixes (Patched in this Fork)
This fork includes essential fixes for Devanagari that are **not available** upstream:
* **Causal Shift Fix**: Fixed the next-token prediction loss in `t3.py`.
* **Tokenizer Logic**: Prevented double-prepending of `[ne]` tags.
* **Alignment Safety**: Increased repetition tolerance from 2 tokens (too aggressive for long vowels) to 15 tokens (~600ms) in `alignment_stream_analyzer.py` to stop early audio cutoffs.

## 📄 License & Credits
* Original architecture by **Resemble AI**.
* Fine-tuning and Nepali optimization by **officialuser**.
* Distributed under the **MIT License**.
