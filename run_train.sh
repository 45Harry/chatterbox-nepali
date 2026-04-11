#!/bin/bash
source .venv/bin/activate
export PYTHONPATH=src
mkdir -p results logs

LOG_FILE="logs/train_$(date +%Y%m%d_%H%M%S).log"
echo "📝 Logging to $LOG_FILE"

python3 src/chatterbox/train_nepali.py \
  --manifest data/train_clean.jsonl \
  --device cuda \
  --batch_size 16 \
  --accum_steps 2 \
  --epochs 30 \
  --save_every 5 \
  --num_workers 4 \
  --resume_t3_weights results/t3_nepali_epoch_25.pt \
  2>&1 | tee "$LOG_FILE"
