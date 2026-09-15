# Scaled 40,000+ Image Physics Deepfake Pipeline (`pipeline_40k`)

This directory contains the production-scaled configuration of the Physics-Based Deepfake Detection Pipeline, designed to stream, cache, and train on **40,000+ images** without memory overflow, network bottlenecks, or loss of progress upon network failures.

## Key Scaling Features

1. **Multi-Worker Resumable Ingestion (`scripts/download_scaled.py`)**:
   - Streams 16,000 Train Real (COCO Train 2017) and 16,000 Train Fake (DiffusionDB).
   - Streams 4,000 Test Real (COCO Val 2017) and 4,000 Test Fake (Held-out Modern Generators).
   - **Atomic Resume**: Scans existing disk files before fetching, allowing instant recovery if interrupted.
   - On-the-fly RGB JPEG validation.

2. **Batched Sharded Feature Caching (`scripts/cache_scaled_features.py`)**:
   - PyTorch `DataLoader` with batching and `torch.no_grad()`.
   - Flushes intermediate tensor shards (`shard_0000.pt`, etc.) every **2,000 processed samples** into `data/cache_scaled/shards/`.
   - Merges shards automatically into `train_scaled_features.pt` ($[32000, 14]$) and `test_scaled_features.pt` ($[8000, 14]$).

3. **40k Scaled Training Engine (`scripts/train_scaled.py`)**:
   - Trains `GatedCrossGenClassifier` with learnable missing-cue mask tokens.
   - Optimizer: `AdamW(lr=5e-4, weight_decay=1e-3)`.
   - Scheduler: `CosineAnnealingLR(T_max=40, eta_min=1e-6)`.
   - Batch size: 256 for 40 epochs.
   - Validation evaluation every epoch, saving best checkpoint to `models/gated_cross_gen_40k_best.pt`.

4. **Observability Evaluation Sweep (`scripts/evaluate_scaled.py`)**:
   - Full evaluation across 8,000 unseen test samples.
   - Automated sweep across $\tau_{\text{obs}} \in [0.5, 2.0]$.

---

## Execution Workflow

```bash
cd g:\Thesis\pipeline_40k

# 1. Download & verify dataset (resumable)
python scripts/download_scaled.py

# 2. Extract & cache features into shards (fault-tolerant)
python scripts/cache_scaled_features.py

# 3. Train on 32k samples for 40 epochs
python scripts/train_scaled.py

# 4. Evaluate on 8k unseen test samples with observability sweep
python scripts/evaluate_scaled.py
```
