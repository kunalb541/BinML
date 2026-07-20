#!/bin/bash
# BinML v5 — unattended overnight training.
#
# Three things make this safe to leave running while the machine is asleep-adjacent:
#   1. `caffeinate -i -m` holds off idle sleep and disk sleep for the life of the run. Without
#      it macOS suspends the process and MPS work dies. (-i = idle, -m = disk. Add -s if the
#      lid will be CLOSED; on Apple silicon a closed lid sleeps regardless of power unless the
#      display-off setting is changed.)
#   2. A supervisor loop restarts the trainer if it exits non-zero, and --resume picks up from
#      the last completed epoch, so an interruption costs one epoch rather than the night.
#   3. Everything is logged with timestamps to a file that can be tailed at any point.
set -u

CACHE="$HOME/Desktop/Research/microlensing/v5mm"
OUT="$HOME/Desktop/Research/microlensing/v5runs/binml_v5_base.pt"
LOG="$HOME/Desktop/Research/microlensing/v5runs/train_$(date +%Y%m%d_%H%M).log"
EPOCHS=${EPOCHS:-18}
BATCH=${BATCH:-512}
MAX_RESTARTS=${MAX_RESTARTS:-5}

mkdir -p "$(dirname "$OUT")"
cd "$(dirname "$0")" || exit 1

echo "=== BinML v5 base training ===" | tee -a "$LOG"
echo "cache=$CACHE" | tee -a "$LOG"
echo "out=$OUT" | tee -a "$LOG"
echo "epochs=$EPOCHS batch=$BATCH" | tee -a "$LOG"
echo "started $(date)" | tee -a "$LOG"

attempt=0
while [ "$attempt" -le "$MAX_RESTARTS" ]; do
  if [ "$attempt" -gt 0 ]; then
    echo "--- restart $attempt/$MAX_RESTARTS at $(date) ---" | tee -a "$LOG"
    RESUME="--resume"
  else
    # Resume is harmless on a fresh run (no .last file) and essential if the user re-runs
    # this script after an interruption, so it is always passed.
    RESUME="--resume"
  fi

  caffeinate -i -m python -u -m pipeline.sim_v5.train_v5 \
      --cache "$CACHE" --out "$OUT" \
      --epochs "$EPOCHS" --batch-size "$BATCH" \
      --lr 4e-4 --d-model 96 --n-layers 4 \
      --device mps --alpha-nonpspl 2.0 $RESUME 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}

  if [ "$rc" -eq 0 ]; then
    echo "=== COMPLETED cleanly at $(date) ===" | tee -a "$LOG"
    exit 0
  fi
  echo "!!! exited rc=$rc at $(date); will resume" | tee -a "$LOG"
  attempt=$((attempt + 1))
  sleep 20
done

echo "=== GAVE UP after $MAX_RESTARTS restarts ===" | tee -a "$LOG"
exit 1
