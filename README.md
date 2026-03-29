# 🇳🇵 Chatterbox-TTS (Nepali Edition)

Fine-tuned text-to-speech for the Nepali language, based on the **Chatterbox-Multilingual-500M** architecture. This repository contains the custom training logic, bug fixes for the Devanagari script, and tools for high-fidelity Nepali voice cloning.

![Nepali TTS Preview](./Chatterbox-Multilingual.png)

## 🚀 Key Features
* **Full Devanagari Support**: Patched tokenizer and alignment analyzer to handle complex Nepali character clusters without cutting off.
* **Seamless Voice Cloning**: High-quality zero-shot cloning of Nepali voices using a 5-10 second reference clip.
* **Optimized for Mac**: Pre-configured for **Apple Silicon (MPS)** acceleration and memory-efficient training on M2/M3 chips.
* **Clean Inference**: Dedicated Gradio UI and test scripts for rapid experimentation.

## 📦 Model Checkpoints
To ensure repository performance, large training checkpoints are hosted on Hugging Face:
- **Repo Link**: [https://huggingface.co/officialuser/chatterbox-nepali](https://huggingface.co/officialuser/chatterbox-nepali)

| File | Purpose | Recommendation |
| :--- | :--- | :--- |
| `t3_nepali_epoch_20.pt` | **Base Nepali Checkpoint** | Use for **quick testing** or as a starting point to **train further**. |

*Place this file in the root folder of this repository after downloading.*

---

## ⚙️ Installation
Follow these steps to set up the environment and download the codebase:

```bash
# 1. Create and activate a dedicated conda environment
conda create -n chatterbox_ne python=3.11 -y
conda activate chatterbox_ne

# 2. Clone the repository
git clone https://github.com/officialuser/chatterbox-nepali.git
cd chatterbox-nepali

# 3. Install dependencies in editable mode
pip install -e .
```

---

## 🏋️ Training / Generation
This project encourages you to build the final model weights yourself! When you reach the end of your training cycle (e.g., 50 epochs), the script will automatically consolidate your efforts and generate a high-performance **`t3_mtl_nepali_final.safetensors`** file.

### How to Resume Training:
```bash
# Make sure your environment is active
conda activate chatterbox_ne
export PYTHONPATH=src

python3 src/chatterbox/train_nepali.py \
  --manifest data/nepali/metadata.csv \
  --device mps \
  --batch_size 4 \
  --accum_steps 4 \
  --epochs 50 \
  --save_every 5 \
  --resume_t3_weights "t3_nepali_epoch_20.pt"
```

---

## 🎙️ Inference & Implementation

### 🛡️ Quick Testing
To generate Nepali speech correctly, you **must** use the provided test scripts. Standard library imports from Hugging Face will not support Devanagari without these specific patches.

#### Using a Checkpoint (.pt):
```bash
# Make sure your environment is active
conda activate chatterbox_ne
export PYTHONPATH=src

python3 test_nepali.py \
  --checkpoint "t3_nepali_epoch_20.pt" \
  --ref_audio "data/nepali/wavs/nep_sample.wav" \
  --text "इन्द्रेणी वा इन्द्रधनुष प्रकाश र रंगबाट उत्पन्न भएको यस्तो घटना हो जसमा रंगीन प्रकाशको एउटा अर्धवृत आकाशमा देखिन्छ। जब सूर्यको प्रकाश पृथ्वीको वायुमण्डलमा भएको पानीको थोपा माथि पर्छ, पानीको थोपाले प्रकाशलाई परावर्तन, आवर्तन र डिस्पर्सन गर्दछ।" \
  --output "nepali_test.wav"
```

#### Using Final Weights (.safetensors):
Once you have trained the model and produced the `.safetensors` file, the command is the same:
```bash
# Make sure your environment is active
conda activate chatterbox_ne
export PYTHONPATH=src

python3 test_nepali.py \
  --checkpoint "t3_mtl_nepali_final.safetensors" \
  --ref_audio "data/nepali/wavs/nep_sample.wav" \
  --text "तपाईंको नयाँ नेपाली एआई तयार छ।" \
  --output "final_output.wav"
```

### 🏮 Web UI (Gradio)
Launch a graphical interface to test voices instantly:
```bash
# Automatically detects and loads local Nepali weights
# Make sure your environment is active
conda activate chatterbox_ne
export PYTHONPATH=src

python3 gradio_nepali.py
```

## 🛠️ Critical Bug Fixes (Patched in this Fork)
This fork includes essential fixes for Devanagari that are **not available** in the original repository:
* **Causal Shift Fix**: Fixed the next-token prediction loss in `t3.py`.
* **Tokenizer Logic**: Prevented double-prepending of `[ne]` tags.
* **Alignment Safety**: Increased repetition tolerance in `alignment_stream_analyzer.py` to stop early audio cutoffs on long Nepali vowels.

## 📄 License & Credits
* Original architecture by **Resemble AI**.
* Fine-tuning and Nepali optimization by **officialuser**.
* Distributed under the **MIT License**.
