#!/usr/bin/env bash
# Detectability-conditioned-labelling ablation: train two identical models on the same data,
# one on observational (shipped) labels and one on generator-intent labels, then evaluate both
# against the observational truth. Demonstrates (rather than merely motivates) contribution #1.
set -euo pipefail
cd "$(dirname "$0")/.."
WORK="${1:-/tmp/binml_ablation}"
NSHARDS="${2:-80}"
SEED=20260720
# The shard bucket is private to the authors' AWS account, so it is supplied by environment
# rather than hard-coded: an S3 URI of this form embeds the account ID, which does not belong in
# a public repository. Point BINML_SHARD_BUCKET at any bucket holding the binned cache shards.
BUCKET="${BINML_SHARD_BUCKET:?set BINML_SHARD_BUCKET to the s3:// prefix holding shard_*.h5}"
mkdir -p "$WORK"/{cache,mm}

echo "[1/5] pull $NSHARDS cache shards from S3"
for i in $(seq 0 $((NSHARDS-1))); do
  f=$(printf "shard_%05d.h5" "$i")
  [ -f "$WORK/cache/$f" ] || aws s3 cp "$BUCKET/$f" "$WORK/cache/$f" --quiet
done
echo "    pulled $(ls "$WORK"/cache/*.h5 | wc -l) shards"

echo "[2/5] cache -> memmap"
python3 -m pipeline.to_memmap --in-dir "$WORK/cache" --out "$WORK/mm"

echo "[3/5] train A (observational labels)"
python3 -m pipeline.train --cache "$WORK/mm" --out "$WORK/model_A_observational.pt" \
  --label-source observational --truncate-aug 0 --epochs 8 --seed $SEED --device mps

echo "[4/5] train B (generator-intent labels -- the ablation)"
python3 -m pipeline.train --cache "$WORK/mm" --out "$WORK/model_B_generator.pt" \
  --label-source generator --truncate-aug 0 --epochs 8 --seed $SEED --device mps

echo "[5/5] evaluate both against observational truth"
python3 validation/ablation_eval.py "$WORK/mm" \
  "$WORK/model_A_observational.pt" "$WORK/model_B_generator.pt" $SEED \
  | tee "$WORK/ablation_result.txt"
echo "done -> $WORK/ablation_result.txt"
