#!/bin/bash
# BinML v5 stage 2: warm restart from the stage-1 epoch-3 checkpoint.
#
# Stage 1 drifted: NonPSPL precision fell 0.765 -> 0.426 over epochs 3-5 while recall rose to
# 0.966, i.e. the model progressively over-called the anomaly class. The cause was
# --alpha-nonpspl 2.0, which gives NonPSPL twice the class mass; early on that lifts recall,
# but once the model can exploit the asymmetry it is optimal under the weighted objective to
# call everything NonPSPL. So: same weights, unbiased class mass, gentler LR.
#
# --init-weights (NOT --resume) is deliberate: it takes the weights and starts a FRESH
# optimiser and OneCycle schedule. Resuming would restore the very optimiser state that was
# driving the drift.
set -u
OUT="$HOME/Desktop/Research/microlensing/v5runs/binml_v5_stage2.pt"
LOG="$HOME/Desktop/Research/microlensing/v5runs/stage2_$(date +%Y%m%d_%H%M).log"
INIT="$HOME/Desktop/Research/microlensing/v5runs/binml_v5_stage1_ep3.pt"
cd "$(dirname "$0")" || exit 1
echo "=== stage 2: warm restart, alpha=1.0, lr=1.2e-4, 8 epochs ===" | tee -a "$LOG"
echo "init from: $INIT" | tee -a "$LOG"
echo "started $(date)" | tee -a "$LOG"
attempt=0
while [ "$attempt" -le 3 ]; do
  [ "$attempt" -gt 0 ] && { echo "--- restart $attempt $(date) ---" | tee -a "$LOG"; R="--resume"; } || R=""
  caffeinate -i -m python -u -m pipeline.sim_v5.train_v5 \
      --cache "$HOME/Desktop/Research/microlensing/v5mm" --out "$OUT" \
      --epochs 8 --batch-size 512 --lr 1.2e-4 --d-model 96 --n-layers 4 \
      --device mps --alpha-nonpspl 1.0 --init-weights "$INIT" $R 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] && { echo "=== COMPLETED $(date) ===" | tee -a "$LOG"; exit 0; }
  echo "!!! rc=$rc at $(date)" | tee -a "$LOG"; attempt=$((attempt+1)); sleep 20
done
exit 1
