---
language:
  - ne
license: mit
tags:
  - text-to-speech
  - nepali
  - voice-cloning
  - tts
  - chatterbox
base_model:
  - ResembleAI/chatterbox
library_name: chatterbox-tts
pipeline_tag: text-to-speech
widget:
  - text: "नमस्ते, म नेपाली एआई हुँ। मलाई तपाईंसँग कुरा गर्न पाउँदा खुसी लागेको छ।"
    example_title: "नेपाली अभिवादन"
  - text: "नेपाल हिमाल, पहाड र तराईले भरिएको सुन्दर देश हो।"
    example_title: "नेपालको भूगोल"
  - text: "काठमाडौं उपत्यकाको ऐतिहासिक र सांस्कृतिक महत्त्व धेरै ठूलो छ।"
    example_title: "काठमाडौंको इतिहास"
---

# 🇳🇵 Chatterbox Nepali TTS

Fine-tuned **Nepali text-to-speech** model based on [Chatterbox-Multilingual-500M](https://huggingface.co/ResembleAI/chatterbox). Supports high-quality zero-shot voice cloning from a short reference clip.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/45Harry/chatterbox-nepali/blob/main/colab_demo.ipynb)

## 🚀 Google Colab

### Option 1: Quick Inference (T4 / Free Tier)

Run the full pipeline directly in Colab. Change runtime type to **T4 GPU** for best results.

#### Cell 1 — Install Dependencies
```python
!pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu121
!git clone https://github.com/45Harry/chatterbox-nepali.git /tmp/cb
!cp /tmp/cb/pyproject_colab.toml /tmp/cb/pyproject.toml
!pip install -q -e /tmp/cb safetensors
```

#### Cell 2 — Load Model
```python
import torch, torchaudio
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from IPython.display import Audio

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load base model
model = ChatterboxMultilingualTTS.from_pretrained(device)

# Load Nepali fine-tuned weights
ckpt = hf_hub_download("Imbatmann/chatterbox-nepali-tts", "t3_mtl_nepali_final.safetensors")
sd = load_file(ckpt)
cleaned = {k.replace("patched_model.", "").replace("model.", ""): v for k, v in sd.items()}
model.t3.load_state_dict(cleaned, strict=False)
model.t3.to(device).eval()
print("✅ Model loaded!")
```

#### Cell 3 — Download Reference Audio
```python
!wget -q https://huggingface.co/Imbatmann/chatterbox-nepali-tts/resolve/main/ref.wav -O ref.wav
# Or upload your own 5-10s reference clip
```

#### Cell 4 — Generate Nepali Speech
```python
text = "नमस्ते, म नेपाली एआई हुँ। मलाई तपाईंसँग कुरा गर्न पाउँदा खुसी लागेको छ।"

wav = model.generate(
    text=text,
    language_id="ne",
    audio_prompt_path="ref.wav",
    exaggeration=0.5,
    temperature=0.8,
)

torchaudio.save("output.wav", wav, model.sr)
print(f"✅ Saved: output.wav ({wav.shape[1]/model.sr:.1f}s)")
Audio("output.wav")
```

#### Cell 5 — Batch Generation (Multiple Texts)
```python
texts = [
    "नेपाल हिमाल, पहाड र तराईले भरिएको सुन्दर देश हो।",
    "काठमाडौं उपत्यकाको ऐतिहासिक र सांस्कृतिक महत्त्व धेरै ठूलो छ।",
    "नेपाली भाषा धेरै मीठो र गम्भीर छ।",
]

for i, txt in enumerate(texts):
    w = model.generate(txt, "ne", audio_prompt_path="ref.wav", exaggeration=0.5, temperature=0.8)
    torchaudio.save(f"batch_{i}.wav", w, model.sr)
    print(f"✅ batch_{i}.wav — {w.shape[1]/model.sr:.1f}s")
    display(Audio(f"batch_{i}.wav"))
```

### Option 2: Gradio Web UI in Colab

```python
# Cell 1
!git clone https://github.com/45Harry/chatterbox-nepali.git
%cd chatterbox-nepali
!pip install -q -e . gradio

# Cell 2
!python gradio_nepali.py --share
# Click the gradio.live link when it appears
```

## Quickstart

```python
# pip install chatterbox-tts
import torch, torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load base model
model = ChatterboxMultilingualTTS.from_pretrained(device)

# Load Nepali fine-tuned weights
ckpt = hf_hub_download("Imbatmann/chatterbox-nepali-tts", "t3_mtl_nepali_final.safetensors")
sd = load_file(ckpt)
cleaned = {k.replace("patched_model.", "").replace("model.", ""): v for k, v in sd.items()}
model.t3.load_state_dict(cleaned, strict=False)
model.t3.to(device).eval()

# Generate Nepali speech
text = "नमस्ते, म नेपाली एआई हुँ। मलाई तपाईंसँग कुरा गर्न पाउँदा खुसी लागेको छ।"
wav = model.generate(
    text=text,
    language_id="ne",
    audio_prompt_path="ref.wav",
    exaggeration=0.5,
    temperature=0.8,
)
ta.save("output-nepali.wav", wav, model.sr)

# Clone a different voice
wav = model.generate(
    text="काठमाडौं उपत्यकाको ऐतिहासिक र सांस्कृतिक महत्त्व धेरै ठूलो छ।",
    language_id="ne",
    audio_prompt_path="YOUR_VOICE.wav",
    exaggeration=0.5,
    temperature=0.8,
)
ta.save("output-cloned.wav", wav, model.sr)
```

## Using the CLI

```bash
# Install
pip install -U chatterbox-tts

# Generate
python -c "
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import torch, torchaudio

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ChatterboxMultilingualTTS.from_pretrained(device)

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
sd = load_file(hf_hub_download('Imbatmann/chatterbox-nepali-tts', 't3_mtl_nepali_final.safetensors'))
cleaned = {k.replace('patched_model.','').replace('model.',''):v for k,v in sd.items()}
model.t3.load_state_dict(cleaned, strict=False)
model.t3.to(device).eval()

wav = model.generate('नमस्ते संसार', 'ne', audio_prompt_path='ref.wav')
torchaudio.save('out.wav', wav, model.sr)
"
```

## Gradio Web UI

```bash
git clone https://github.com/45Harry/chatterbox-nepali.git
cd chatterbox-nepali
pip install -e .
python gradio_nepali.py
```

## Training Your Own

```bash
# 1. Prepare dataset (pipe-separated CSV)
# data/train.jsonl: {"audio_path": "wavs/001.wav", "text": "नमस्ते संसार"}

# 2. Run training
bash run_train.sh
# or directly:
python src/chatterbox/train_nepali.py \
  --manifest data/train.jsonl \
  --device cuda \
  --batch_size 16 \
  --accum_steps 2 \
  --epochs 50 \
  --save_every 5 \
  --resume_t3_weights results/t3_nepali_epoch_25.pt
```

## Model Details

| Parameter | Value |
|---|---|
| Architecture | Token-to-Token Transformer (LLaMA 520M) |
| Languages | Nepali (`ne`) |
| Sample Rate | 24,000 Hz |
| Frame Rate | 25 Hz (speech tokens) |
| Vocoder | S3Gen (CFM + HiFiGAN) |

## Features

- **Devanagari Support** — Full Nepali script handling with NFKD normalization
- **Zero-shot Voice Cloning** — Clone any voice from 5-10s reference audio
- **Emotion Control** — Exaggeration parameter (0.0-1.0) for pacing/style
- **Gradio UI** — Built-in web interface for easy testing

## Sample Output

Listen to the generated Nepali speech samples in the **Files and versions** tab.

## License

MIT License — Original architecture by Resemble AI, Nepali fine-tuning by Imbatmann.
