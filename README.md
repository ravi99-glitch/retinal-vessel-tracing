# Retinal Vessel Tracing

A reinforcement learning agent for extracting the centerline of retinal blood vessels. The agent was compared against classical and CNN baseline models using multiple datasets.

## Overview

This project trains an RL agent to trace blood vessel centerlines in retinal fundus images. The agent learns to navigate along blood vessel structures using a policy, trained via imitation learning followed by PPO fine-tuning. A seed detector predicts starting points, allowing for end-to-end inference without the need for ground-truth labels.

The agent was compared against three baseline models: a Frangi vesselness filter, a greedy tracing heuristic, and a UNet CNN.


## Datasets

Training and validation use a balanced combination of five datasets. Testing uses two external datasets the models never see during training.

| Split | Datasets | Purpose |
|-------|----------|---------|
| Train / Val | FIVES, STARE, CHASE_DB1, HRF, LES-AV | Model training with weighted sampling for balance |
| Test | DRIVE, DR_HAGIS | Generalization evaluation on unseen data |


## Methods

### Baseline Models

**Frangi filter** — Multi-scale Hessian-based vesselness enhancement followed by thresholding and skeletonization. Scale parameters are tuned per test dataset.

**Greedy tracer** — Seeds placed at local vesselness maxima, traces grown greedily along ridges, then skeletonized and pruned.

**UNet CNN** — Lightweight UNet trained on CLAHE-preprocessed grayscale images to predict vessel centerline probability maps.

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

---
*This document was created with assistance from AI tools. The content has been reviewed and edited by the project author.*
