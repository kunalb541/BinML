# Training pipeline

> **v5 note.** The current 6-class multi-band pipeline — simulation, binning, training,
> fine-tuning, evaluation, and every module's role — is documented in
> **[`pipeline_v5.md`](pipeline_v5.md)**, with verified quick-start commands. Read that.
>
> This page is retained for the **legacy 3-class CNN-GRU** pipeline (`binml/` package) only.

---

## v5 training in one screen

Full detail and file-by-file in [`pipeline_v5.md`](pipeline_v5.md); the short version:

```bash
# 1. simulate raw shards (all 6 classes; --regime for hard corners; --seed-base for unseen sets)
python -m pipeline.sim_v5.run_shard --shard 0 --n-shards 1 --out data/raw

# 2. bin to a compact cache (local)
python -c "from pipeline.sim_v5.cache import build_cache; import glob; \
build_cache(sorted(glob.glob('data/raw/*.h5')), 'data/cache/shard_00000.h5')"

# 3. cache -> shuffled fp16 memmap
python -m pipeline.sim_v5.to_memmap --in-dir data/cache --out data/mm

# 4a. train from scratch
python -m pipeline.sim_v5.train_v5 --cache data/mm --out runs/model.pt --epochs 6 --device mps

# 4b. OR warm-start fine-tune with the real-time cascade enabled
python -m pipeline.sim_v5.train_v5 --cache data/mm --out runs/stage6.pt \
  --init-weights runs/stage5.pt --truncate-aug 0.5 --alpha-nonpspl 1.0 \
  --lr 5e-5 --epochs 6 --resume
```

### The two levers that matter

- **`--truncate-aug 0.5`** turns on truncation augmentation with **detectability-conditioned
  relabelling**: a partially-observed season is relabelled by what is observable in the revealed
  window (undetectable → Flat; binary before its `t_anom` → PSPL). This is what produces the
  real-time cascade — it is the single most important flag for the v5 behaviour.

- **Warm-start fine-tuning beats more base training.** The model line is
  `base → stage2 → … → stage6`, each a targeted warm-start (`--init-weights`) on harder data
  (hard microlensing regimes, then the cascade + weak-spot coverage). Base training plateaued;
  targeted fine-tuning on the hard regimes is the lever.

### Class balancing

`train_v5.compute_weights` class-weights the 6-way cross-entropy, so a Flat-heavy training mix
doesn't bias the loss. Weighting undoes both the natural class imbalance and the byproduct
subsampling.

For the honest evaluation of the resulting checkpoint, see [`evaluation.md`](evaluation.md).
