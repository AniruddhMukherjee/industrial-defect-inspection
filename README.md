# Automated Visual Defect Inspection (Industrial QA)

An anomaly detection and localization system for industrial visual quality inspection, built on the **MVTec AD** benchmark — the standard academic/industrial dataset for this problem, created by a German industrial imaging company. The system flags defective products and shows *where* the defect is, trained using only defect-free images, mirroring how real manufacturing QA actually works: defects are rare and varied, so you can never collect enough labeled examples of every possible flaw.

**Live demo:** [TODO: Hugging Face Spaces link]

![Carpet heatmap example](outputs/heatmap_carpet.png)

## Why this project

Every target manufacturing company — automotive, precision components, sheet-metal, pneumatics — runs visual QA on its production lines, manually or semi-automatically. This project demonstrates an automated visual defect detection + localization system benchmarked against a real industrial dataset, not a generic image classification exercise.

## Approach

**Anomaly detection, not classification.** The model never sees a single defective image during training — only defect-free ("good") units. This matches the real constraint in manufacturing: you have plenty of normal units, but defects are rare, varied, and expensive to label exhaustively.

**Method: PatchCore-style embedding + nearest-neighbor distance.**
1. A frozen, ImageNet-pretrained WideResNet-50 extracts patch-level features (activations from `layer2` + `layer3`, giving a 28×28 grid of 1536-dim feature vectors per image — not a single whole-image descriptor).
2. Every normal training image's patch features are pooled into one large "memory bank" representing what normal looks like.
3. The bank is reduced via **greedy coreset subsampling** (farthest-point / greedy k-center selection) down to ~1% of its original size, keeping the most representative points instead of redundant near-duplicates from repetitive texture.
4. At inference, each patch of a new image is compared to its nearest neighbor in the bank. The image-level anomaly score is the **max** distance across all patches (one bad patch is enough to flag the image). The full grid of per-patch distances, upsampled to image resolution, becomes the **defect localization heatmap**.

No backpropagation happens anywhere in this pipeline — the backbone is frozen, and the "model" is literally the stored memory bank plus a distance lookup, not a set of learned weights.

### Why PatchCore over a reconstruction autoencoder

A convolutional autoencoder (reconstruct normal images, flag high reconstruction error as anomalous) was implemented and evaluated as a comparison baseline. It scored **0.3848 AUROC on carpet**, worse than random guessing, versus PatchCore's 0.998. Visual inspection of reconstructions explains why: a well-trained autoencoder on repetitive texture learns general "carpet-like" patterns and partially reconstructs dark/irregular blobs as locally plausible texture, smoothing out the pixel-error signal exactly where it's needed most. This is a documented weakness of reconstruction-based anomaly detection, and it's why PatchCore-style memory-bank comparison was chosen as the primary method.

(Getting the autoencoder itself to train correctly also surfaced a real optimization bug worth noting: with a bias-initialized final layer, a standard learning rate (1e-3) caused Adam to immediately overshoot a good starting point in the first few gradient steps, plateauing the model at a *worse* loss (~0.41) than a trivial constant-color baseline (~0.013) for dozens of epochs. Lowering the learning rate 10x (to 1e-4) and adding gradient clipping fixed the instability, dropping final loss to ~0.002 — after which the reconstruction-quality limitation above became visible and measurable.)

## Dataset

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) — 15 object/texture categories, ~5,354 images, CC BY-NC-SA 4.0 (non-commercial use with attribution).

This project uses two categories to demonstrate the approach generalizes across fundamentally different visual structure:
- **carpet** (texture category) — 280 train / 117 test images
- **bottle** (object category) — 209 train / 83 test images

## Results

### Image-level detection (AUROC)

| Category | PatchCore AUROC | Autoencoder AUROC |
|----------|-----------------|--------------------|
| carpet   | **0.9980**      | 0.3848             |
| bottle   | **0.9992**      | not evaluated      |

### Pixel-level localization

| Category | Pixel AUROC | Best IoU (oracle threshold) | Defective pixel fraction |
|----------|-------------|------------------------------|---------------------------|
| carpet   | 0.9893      | 0.4753                       | 2.04%                     |
| bottle   | 0.9802      | 0.6014                       | 7.52%                     |

**Why pixel AUROC and IoU disagree in ranking:** AUROC is a ranking metric, unaffected by defect size — carpet's small, sharp defects against uniform texture are easy to rank correctly, giving it the higher AUROC. IoU measures spatial overlap at a fixed threshold, and is sensitive to defect size relative to the model's fixed localization resolution (a 28×28 patch grid upsampled to 224×224). Bottle's defects are proportionally larger, so the same absolute localization imprecision costs it less in IoU terms than it costs carpet, despite bottle's slightly lower AUROC. Both metrics are reported because they measure genuinely different things, and relying on only one would be misleading.

### Live demo thresholds

Per-category thresholds were calibrated as the midpoint between the highest "good" test score and lowest "defective" test score, after an initial calibration attempt (99th percentile of *training*-set scores) proved too strict — training images score artificially low against a memory bank partly built from their own patches, which doesn't represent how a genuinely new normal image scores. The corrected thresholds:

| Category | Threshold |
|----------|-----------|
| carpet   | 3.806     |
| bottle   | 4.3088    |

## Repository structure
