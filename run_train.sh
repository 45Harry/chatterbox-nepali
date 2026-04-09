#!/bin/bash
source .venv/bin/activate
export PYTHONPATH=src
python3 src/chatterbox/train_nepali.py \
  --manifest data/train_clean.jsonl \
  --device cuda \
  --batch_size 16 \
  --epochs 30 \
  --save_every 5 \
  --num_workers 4 \
  --resume_t3_weights base_model/t3_nepali_epoch_20.pt
