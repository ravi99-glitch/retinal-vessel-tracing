# Retinal Vessel Tracing

Reinforcement learning agent for retinal vessel centerline extraction, benchmarked against classical and CNN baselines on multi-dataset evaluation.

https://lh3.googleusercontent.com/gg-dl/AOI_d_94oYbSiWvFT7jEjhTndeYyr04ruPV6idSVyb8K_lWQmZ-pa55r2jl6clEyXw3z-TEM5IN7M9mDM6QUrgWBB0NHJ4X9OtOf6RP1uGtgydZhds4F-uSgGKmhQcpGChhzWXRUItwpabOWOfgHwjNNQzEIIAvmCVop8aHtJ2GlxzgLsRUtag=s1600-rj<img width="1408" height="768" alt="image" src="https://github.com/user-attachments/assets/87c0fda4-7538-4bab-a92c-c2e2b03eb715" />


## Overview

This project trains an RL agent to trace blood vessel centerlines in retinal fundus images. The agent learns to navigate along vessel structures using a policy trained via imitation learning followed by PPO fine-tuning. A seed detector predicts starting points, enabling fully end-to-end inference without ground-truth dependencies.

Three baselines provide comparison: a Frangi vesselness filter, a greedy tracing heuristic, and a UNet CNN.

## Pipeline

```
                    ┌─────────────────────────────────────┐
                    │         Training (5 datasets)        │
                    │   DRIVE · STARE · CHASE_DB1 · HRF   │
                    │              · LES-AV                │
                    └──────────────────┬──────────────────┘
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            │                          │                          │
     ┌──────▼──────┐          ┌───────▼────────┐         ┌──────▼──────┐
     │  UNet CNN   │          │  Seed Detector  │         │  Imitation  │
     │  Baseline   │          │   (heatmap)     │         │  Learning   │
     └──────┬──────┘          └───────┬────────┘         └──────┬──────┘
            │                         │                         │
            │                         │                  ┌──────▼──────┐
            │                         │                  │     PPO     │
            │                         │                  │ Fine-tuning │
            │                         │                  └──────┬──────┘
            │                         │                         │
            │                  ┌──────▼─────────────────────────▼──────┐
            │                  │        End-to-End Inference           │
            │                  │  Seeds → Frontier Tracer → Skeleton   │
            │                  └──────────────────┬───────────────────┘
            │                                     │
            └──────────────┬──────────────────────┘
                           │
                    ┌──────▼──────────────────────────────┐
                    │        Evaluation (2 datasets)       │
                    │         AV-WIDE · DR_HAGIS           │
                    └─────────────────────────────────────┘
```

## Datasets

Training and validation use a balanced combination of five datasets. Testing uses two held-out external datasets the models never see during training.

| Split | Datasets | Purpose |
|-------|----------|---------|
| Train / Val | DRIVE, STARE, CHASE_DB1, HRF, LES-AV | Model training with weighted sampling for balance |
| Test | AV-WIDE, DR_HAGIS | Generalization evaluation on unseen data |


## Methods

### Baselines

**Frangi filter** — Multi-scale Hessian-based vesselness enhancement followed by thresholding and skeletonization. Scale parameters are tuned per test dataset.

**Greedy tracer** — Seeds placed at local vesselness maxima, traces grown greedily along ridges, then skeletonized and pruned.

**UNet CNN** — Lightweight UNet (DSConv blocks, ~0.5M params) trained on CLAHE-preprocessed grayscale images to predict vessel centerline probability maps.

### RL Agent

The RL agent treats centerline extraction as sequential decision-making. At each step, the agent observes a local 65×65 patch and chooses one of 8 movement directions.

**Training pipeline:**
1. **Seed detector** — UNet trained to predict a heatmap of vessel endpoints and junctions
2. **Imitation learning** — Policy initialized by supervised learning on expert traces derived from ground-truth centerlines
3. **PPO fine-tuning** — Policy refined with proximal policy optimization using a shaped reward combining coverage, proximity, and topology signals

**Inference:**
1. Seed detector predicts starting points from the input image
2. Frontier tracer sequentially launches the RL agent from each seed
3. Individual traces are merged into a single skeleton

## Metrics

All methods are evaluated using the same metric suite:

| Metric | What it measures |
|--------|-----------------|
| F1 @ 1/2/3 px | Tolerance-aware centerline matching |
| Precision / Recall | Directional centerline accuracy |
| clDice | Topology-aware volumetric overlap |
| IoU | Binary segmentation overlap |
| HD95 | 95th percentile Hausdorff distance |
| Betti-0 error | Connected component count difference |

Results are saved as per-image CSVs and summary tables under `results/`.
