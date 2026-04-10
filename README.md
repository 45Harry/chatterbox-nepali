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

> [!TIP]
> While the provided checkpoint works for testing, we recommend training for **50 to 100 epochs** on your dedicated dataset to achieve significantly more natural-sounding voices and smoother Nepali prosody.

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

# 4. (Optional) For GPU training with CUDA, reinstall PyTorch with GPU support:
#    PyPI doesn't host CUDA-enabled wheels by default, so install from the CUDA index:
pip uninstall -y torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> [!NOTE]
> Updated `pyproject.toml` to use `torch>=2.11.0` and `torchaudio>=2.11.0` for Python < 3.14.

---

## 🎙️ Inference & Implementation

### 🛡️ Quick Testing
To generate Nepali speech correctly, you **must** use the provided test scripts. Standard library imports from Hugging Face will not support Devanagari without these specific patches.

#### 🎧 Hear the Results
**[Listen to the Nepali Rainbow Benchmark Output](./nepali_test_output.wav)**

#### Benchmark (M2 Max 64GB)
- **Input**: Long paragraph (~45 words)
- **Reference Audio**: 10 seconds (samples/achyut_ref_10s.wav)
- **Generation Time (MPS)**: **~115 seconds**
- **Real-time Factor**: ~0.35x

#### Run the test command:
```bash
# Make sure your environment is active
conda activate chatterbox_ne
export PYTHONPATH=src

python3 test_nepali.py \
  --checkpoint "t3_nepali_epoch_20.pt" \
  --ref_audio "samples/achyut_ref_10s.wav" \
  --text "इन्द्रेणी वा इन्द्रधनुष प्रकाश र रंगबाट उत्पन्न भएको यस्तो घटना हो जसमा रंगीन प्रकाशको एउटा अर्धवृत आकाशमा देखिन्छ। जब सूर्यको प्रकाश पृथ्वीको वायुमण्डलमा भएको पानीको थोपा माथि पर्छ, पानीको थोपाले प्रकाशलाई परावर्तन, आवर्तन र डिस्पर्सन गर्दछ।" \
  --output "nepali_test_output.wav"
```

> [!NOTE]
> The file **`nepali_test_output.wav`** in the root of this repository was generated using the command above. For testing emotional expressiveness and paralinguistic control, we recommend using the **Gradio Interface**. Note: The current Gradio script is optimized for the final `.safetensors` model; you can modify `gradio_nepali.py` if you wish to host a raw `.pt` checkpoint instead.

### 🏮 Web UI (Gradio)
Launch a graphical interface to test voices instantly. It will automatically detect and load your local Nepali weights:
```bash
conda activate chatterbox_ne
export PYTHONPATH=src
python3 gradio_nepali.py
```

---

## 🏋️ Training / Dataset Format
If you wish to fine-tune the model further or use your own voice data, ensure your dataset follows the standard format:

### 1. File Structure
```text
data/nepali/
├── metadata.csv
└── wavs/
    ├── voice_01.wav
    ├── voice_02.wav
    └── ...
```

### 2. `metadata.csv` (Pipe-separated)
The file should **not** have a header. Use the `filename|[ne]text` format:
```csv
voice_01|[ne]नमस्ते संसार यो मेरो आवाज हो
voice_02|[ne]नेपाली भाषा धेरै मीठो छ
```

### 3. Audio Requirements
*   **Format**: Mono WAV (24,000 Hz or 48,000 Hz recommended).
*   **Duration**: 2 to 10 seconds per clip.

### 4. Resume Training
```bash
conda activate chatterbox_ne
export PYTHONPATH=src

python3 src/chatterbox/train_nepali.py \
  --manifest data/nepali/manifest.jsonl \
  --device cuda \
  --batch_size 4 \
  --accum_steps 4 \
  --epochs 50 \
  --save_every 5 \
  --resume_t3_weights results/t3_nepali_epoch_45.pt
```

---

## 🎯 Post-Training (Safetensors Generation)
Once your training reaches the final epoch (e.g. 50), the script will automatically consolidate your efforts and generate a high-performance **`t3_mtl_nepali_final.safetensors`** file.

### Using your Optimized Safetensors:
Loading the finished `.safetensors` format is significantly **faster** and more secure than resuming from `.pt` checkpoints. You can use it in production with the following logic:

```bash
# Use the same test script but point to your final safetensors
python3 test_nepali.py \
  --checkpoint "t3_mtl_nepali_final.safetensors" \
  --ref_audio "your_reference.wav" \
  --text "तपाईंको नयाँ नेपाली एआई तयार छ।" \
  --output "production_output.wav"
```

---

## 🛠️ Critical Bug Fixes (Patched in this Fork)
This fork includes essential fixes for Devanagari that are **not available** in the original repository:
* **Causal Shift Fix**: Fixed the next-token prediction loss in `t3.py`.
* **Tokenizer Logic**: Prevented double-prepending of `[ne]` tags.
* **Alignment Safety**: Increased repetition tolerance in `alignment_stream_analyzer.py` to stop early audio cutoffs on long Nepali vowels.

## 📄 License & Credits
* Original architecture by **Resemble AI**.
* Fine-tuning and Nepali optimization by **Harry & Team**.
* Reference Audio in samples by **Khusiyali Moktan**.
* Distributed under the **MIT License**.



python3 src/chatterbox/train_nepali.py \
  --manifest data/nepali/manifest.jsonl \
  --epochs 50 \
  --batch_size 16 \
  --device cuda

