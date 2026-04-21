---
language:
  - ne
license: mit
tags:
  - text-to-speech
  - nepali
  - chatterbox
  - quantized
  - fp16
base_model:
  - ResembleAI/chatterbox
pipeline_tag: text-to-speech
---

# Chatterbox Nepali Finetuned (FP16 Quantized)

This repository contains FP16 quantized weights for a Nepali fine-tuned Chatterbox TTS setup.

## Included files

- `t3_mtl23ls_v2.safetensors`
- `s3gen.pt`
- `ve.pt`
- `Cangjie5_TC.json`
- `grapheme_mtl_merged_expanded_v1.json`

## Notes

- Intended for Nepali (`ne`) text-to-speech workflows based on Chatterbox.
- Model artifacts are provided as-is for inference use.
