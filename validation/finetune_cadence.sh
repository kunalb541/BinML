#!/usr/bin/env bash
# Fine-tune BinML with cadence augmentation so it degrades gracefully to sparse cadence.
#
# The base model, trained only on Roman's dense grid, misreads a sparsely-sampled light curve as
# a variable star (validation/cadence_robustness.py). The --cadence-aug flag (pipeline/train.py)
# thins events to a random density during training and relabels by what survives, teaching correct
# behaviour at ground-survey cadence. This script warm-starts from the shipped weights and writes a
# NEW checkpoint (it does NOT overwrite binml/weights/binml.pt).
#
# Two data paths:
#   (A) LOCAL  -- regenerate a training set here (no AWS). Fast to simulate (~2000 evt/s) but a
#                 proof-of-concept scale; edit N_SHARDS. Good for confirming the augmentation helps.
#   (B) S3     -- pull the original multi-million-event cache (needs `aws login`; the full retrain).
#
# Usage:  bash validation/finetune_cadence.sh [work_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
WORK="${1:-/tmp/binml_cadence}"
N_SHARDS=16            # local proof-of-concept: ~16 x 7.5k = ~120k events
mkdir -p "$WORK"/{raw,cache,mm}

echo "[1/5] simulate $N_SHARDS shards"
for i in $(seq 0 $((N_SHARDS-1))); do
  python3 -m pipeline.run_shard --shard "$i" --n-shards 200 --out "$WORK/raw"
done

echo "[2/5] bin -> cache"
python3 - "$WORK" <<'PY'
import sys, glob
from pipeline.cache import build_cache
w=sys.argv[1]
for h5 in sorted(glob.glob(f"{w}/raw/*.h5")):
    build_cache([h5], f"{w}/cache/"+h5.split("/")[-1])
PY

echo "[3/5] cache -> memmap"
python3 -m pipeline.to_memmap --in-dir "$WORK/cache" --out "$WORK/mm"

echo "[4/5] fine-tune from shipped weights, with cadence augmentation"
python3 -m pipeline.train \
  --cache "$WORK/mm" --out "$WORK/binml_cadence.pt" \
  --init-weights binml/weights/binml.pt \
  --epochs 4 --cadence-aug 0.5 --truncate-aug 0.5 --device mps

echo "[5/5] re-run the cadence sweep against the new checkpoint"
echo "    edit validation/cadence_robustness.py to point Classifier at $WORK/binml_cadence.pt,"
echo "    or: python -c \"import binml; binml.Classifier(weights='$WORK/binml_cadence.pt')\""
echo "done -> $WORK/binml_cadence.pt"
