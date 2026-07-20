#!/bin/bash
# BinML v5 stage 3 — fine-tune from the stage-2 best checkpoint.
#
# Two goals, both identified from the diagnostic plots rather than guessed:
#  1. EARLY DETECTION. The base model saw F146 100% observed in every training curve, so a
#     partial season is out-of-distribution: at 25% revealed it predicts Eruptive for 100% of
#     events at 0.985 confidence. --truncate-aug 0.5 reveals a random prefix half the time,
#     which is the only way online detection becomes possible at all.
#  2. The weak regimes the parameter-dependence plot exposed: q ~ 2e-5 (recall 0.67),
#     s > 3 wide binaries (0.59), tE > 50 d (0.76).
#
# Low LR and few epochs: this is a fine-tune, not a retrain, and the base is already at
# NonPSPL F1 0.940. Judge it on the change in AVERAGE PRECISION over the whole PR curve, not
# recall at argmax -- otherwise sliding along the existing trade-off looks like improvement.
set -u
OUT="$HOME/Desktop/Research/microlensing/v5runs/binml_v5_stage3.pt"
LOG="$HOME/Desktop/Research/microlensing/v5runs/stage3_$(date +%Y%m%d_%H%M).log"
INIT="$HOME/Desktop/Research/microlensing/v5runs/binml_v5_stage2.pt"
cd "$(dirname "$0")" || exit 1
echo "=== stage 3: fine-tune, truncation aug 0.5, lr 3e-5, 6 epochs ===" | tee -a "$LOG"
echo "init: $INIT" | tee -a "$LOG"; echo "started $(date)" | tee -a "$LOG"
attempt=0
while [ "$attempt" -le 3 ]; do
  [ "$attempt" -gt 0 ] && R="--resume" || R=""
  caffeinate -i -m python -u -m pipeline.sim_v5.train_v5 \
      --cache "$HOME/binml_data/v5mm_train" --out "$OUT" \
      --epochs 6 --batch-size 512 --lr 3e-5 --d-model 96 --n-layers 4 \
      --device mps --alpha-nonpspl 1.0 --truncate-aug 0.5 \
      --init-weights "$INIT" $R 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] && { echo "=== COMPLETED $(date) ===" | tee -a "$LOG"; exit 0; }
  echo "!!! rc=$rc" | tee -a "$LOG"; attempt=$((attempt+1)); sleep 20
done
exit 1
