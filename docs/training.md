# Training pipeline

> **Current model.** This page summarises the 6-class multi-band training commands. The complete
> simulation-to-evaluation walkthrough is in **[`pipeline.md`](pipeline.md)**. The superseded
> 3-class CNN-GRU is documented separately in **[`legacy_3class.md`](legacy_3class.md)** and lives
> under `binml.legacy`.

---

## v5 training in one screen

Full detail and file-by-file in [`pipeline.md`](pipeline.md); the short version:

```bash
# 1. simulate raw shards (all 6 classes; --regime for hard corners; --seed-base for unseen sets)
python -m pipeline.run_shard --shard 0 --n-shards 1 --out data/raw

# 2. bin to a compact cache (local)
python -c "from pipeline.cache import build_cache; import glob; \
build_cache(sorted(glob.glob('data/raw/*.h5')), 'data/cache/shard_00000.h5')"

# 3. cache -> shuffled fp16 memmap
python -m pipeline.to_memmap --in-dir data/cache --out data/mm

# 4a. train from scratch
python -m pipeline.train --cache data/mm --out runs/binml.pt --epochs 6 --device mps

# 4b. OR warm-start fine-tune with partial-season truncation augmentation
python -m pipeline.train --cache data/mm --out runs/binml.pt \
  --init-weights runs/base.pt --truncate-aug 0.5 --alpha-nonpspl 1.0 \
  --lr 5e-5 --epochs 6 --resume
```

### The two levers that matter

- **`--truncate-aug 0.5`** turns on truncation augmentation with **detectability-conditioned
  relabelling**: a partially revealed season is relabelled under the synthetic policy
  (undetectable → Flat; binary before its `t_anom` → PSPL). `t_anom` is derived from the injected,
  noise-free binary-versus-PSPL deviation and is not observable to a live broker. The flag defines
  the intended partial-season label progression; it does not by itself establish a real-time alert
  system or an absence of premature crossings.

- **Reported training lineage.** The shipped checkpoint followed a curriculum of targeted
  warm-starts (`--init-weights`) on harder data and then partial-season augmentation. The retained
  fine-tuning comparisons use one seed per arm and do not provide a matched, repeated-seed test
  against additional base training. Treat them as lineage/provenance, not an established recipe
  advantage.

The released training data use a legacy one-season Roman-like schedule and known outdated
colour-band photometric constants; see [`model_card.md`](model_card.md). A production retrain should
use the audited current constants and preserve a source/config/checkpoint manifest.

### Class balancing

`train.compute_weights` class-weights the 6-way cross-entropy, so a Flat-heavy training mix
doesn't bias the loss. Weighting undoes both the natural class imbalance and the byproduct
subsampling.

For the honest evaluation of the resulting checkpoint, see [`evaluation.md`](evaluation.md).
