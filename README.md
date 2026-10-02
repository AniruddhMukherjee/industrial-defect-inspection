# Automated Visual Defect Inspection (Industrial QA)

An anomaly detection and localization system for industrial visual quality inspection, built on the **MVTec AD** benchmark — the standard academic/industrial dataset for this problem, created by a German industrial imaging company. The system flags defective products and shows *where* the defect is, trained using only defect-free images, mirroring how real manufacturing QA actually works: defects are rare and varied, so you can never collect enough labeled examples of every possible flaw.

**Live demo:** [industrial-image-defect-inspection.streamlit.app](https://industrial-image-defect-inspection.streamlit.app)

![Carpet heatmap example](outputs/heatmap_carpet.png)

## Why this project

Every target manufacturing company — automotive, precision components, sheet-metal, pneumatics — runs visual QA on its production lines, manually or semi-automatically. This project demonstrates an automated visual defect detection + localization system benchmarked against a real industrial dataset, not a generic image classification exercise.

## Approach

**Anomaly detection, not classification.** The model never sees a single defective image during training — only defect-free ("good") units. This matches the real constraint in manufacturing: you have plenty of normal units, but defects are rare, varied, and expensive to label exhaustively.

**Method: PatchCore-style embedding + nearest-neighbor distance.**
1. A frozen, ImageNet-pretrained WideResNet-50 extracts patch-level features (activations from `layer2` + `layer3`, giving a 28×28 grid of 1536-dim feature vectors per image — not a single whole-image descriptor).
2. Every normal training image's patch features are pooled into one large "memory bank" representing what normal looks like.
3. The bank is reduced via **greedy coreset subsampling** (farthest-point / greedy k-center selection) down to a small, representative fraction of its original size, keeping the most informative points instead of redundant near-duplicates from repetitive texture.
4. At inference, each patch of a new image is compared to its nearest neighbor in the bank. The image-level anomaly score is the **max** distance across all patches (one bad patch is enough to flag the image). The full grid of per-patch distances, upsampled to image resolution, becomes the **defect localization heatmap**.

No backpropagation happens anywhere in this pipeline — the backbone is frozen, and the "model" is literally the stored memory bank plus a distance lookup, not a set of learned weights.

### Why PatchCore over a reconstruction autoencoder

A convolutional autoencoder (reconstruct normal images, flag high reconstruction error as anomalous) was implemented and evaluated as a comparison baseline. It scored **0.3848 AUROC on carpet**, worse than random guessing, versus PatchCore's 0.998. Visual inspection of reconstructions explains why: a well-trained autoencoder on repetitive texture learns general "carpet-like" patterns and partially reconstructs dark/irregular blobs as locally plausible texture, smoothing out the pixel-error signal exactly where it's needed most. This is a documented weakness of reconstruction-based anomaly detection, and it's why PatchCore-style memory-bank comparison was chosen as the primary method.

(Getting the autoencoder itself to train correctly also surfaced a real optimization bug worth noting: with a bias-initialized final layer, a standard learning rate (1e-3) caused Adam to immediately overshoot a good starting point in the first few gradient steps, plateauing the model at a *worse* loss (~0.41) than a trivial constant-color baseline (~0.013) for dozens of epochs. Lowering the learning rate 10x (to 1e-4) and adding gradient clipping fixed the instability, dropping final loss to ~0.002 — after which the reconstruction-quality limitation above became visible and measurable.)

## Dataset

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) — 15 object/texture categories, ~5,354 images, CC BY-NC-SA 4.0 (non-commercial use with attribution).

This project evaluates three categories, chosen to show the approach generalizes across fundamentally different visual structure:
- **carpet** (texture category) — 280 train / 117 test images
- **bottle** (object category) — 209 train / 83 test images
- **wood** (texture category) — 247 train / 79 test images

## Results

### Image-level detection (AUROC)

| Category | PatchCore AUROC | Autoencoder AUROC |
|----------|-----------------|--------------------|
| carpet   | **0.9980**      | 0.3848             |
| bottle   | **0.9992**      | not evaluated      |
| wood     | **0.9877**      | not evaluated      |

### Pixel-level localization

| Category | Pixel AUROC | Best IoU (oracle threshold) | Defective pixel fraction |
|----------|-------------|------------------------------|---------------------------|
| carpet   | 0.9893      | 0.4753                       | 2.04%                     |
| bottle   | 0.9802      | 0.6014                       | 7.52%                     |
| wood     | 0.9483      | 0.3391                       | not separately logged     |

**Why pixel AUROC and IoU disagree in ranking:** AUROC is a ranking metric, unaffected by defect size — carpet's small, sharp defects against uniform texture are easy to rank correctly, giving it the higher AUROC. IoU measures spatial overlap at a fixed threshold, and is sensitive to defect size relative to the model's fixed localization resolution (a 28×28 patch grid upsampled to 224×224). Bottle's defects are proportionally larger, so the same absolute localization imprecision costs it less in IoU terms than it costs carpet, despite bottle's slightly lower AUROC. Both metrics are reported because they measure genuinely different things, and relying on only one would be misleading.

### Live demo thresholds

Per-category thresholds are calibrated as the midpoint between the highest "good" test score and lowest "defective" test score, after an initial calibration attempt (99th percentile of *training*-set scores) proved too strict — training images score artificially low against a memory bank partly built from their own patches, which doesn't represent how a genuinely new normal image scores. All metrics and thresholds are computed by a single script (`scripts/finalize_category.py`) and stored in `outputs/category_config.json`, which the dashboard reads directly — nothing is hardcoded or manually retyped.

| Category | Threshold |
|----------|-----------|
| carpet   | 3.806     |
| bottle   | 4.3088    |
| wood     | 4.1929    |

## Interactive dashboard

The Streamlit app lets you pick a product category and upload an image to get a live anomaly score, a localization heatmap, and a DEFECTIVE/NORMAL verdict against that category's calibrated threshold.

- **Main page:** category selector and image upload — the actual task.
- **Sidebar:** read-only model stats for whichever category is selected (AUROC, IoU, dataset size, threshold) — informational, not something you interact with to run inference.

The dropdown, memory bank, and threshold are all driven entirely by `outputs/category_config.json` — there is no hardcoded list of categories or thresholds anywhere in the app code. Adding a category to the config makes it appear in the dropdown automatically.

**Important caveat, confirmed by testing:** the system is sensitive to photographic conditions, not just product type. A photo of carpet pulled from a random website (different lighting, camera, and possibly weave/fiber type than MVTec AD's own photos) gets flagged as anomalous even with no real defect — because every patch looks "unfamiliar" relative to the memory bank, which only ever learned MVTec AD's specific photographic setup. This mirrors real deployments: a factory inspection camera is fixed in place, photographing the same product under the same lighting every time — this is a built-in scope boundary of the approach, not a flaw unique to this implementation.

## Repository structure

```
industrial-defect-inspection/
  src/
    data/dataset.py        # MVTec-style dataset loader (train/test, image+mask pairing)
    models/
      backbone.py          # Frozen WideResNet-50 patch feature extractor
      memory_bank.py        # Feature collection, coreset selection, scoring, heatmap generation
      autoencoder.py         # Comparison baseline
    eval/visualize.py         # Sanity-check and overlay visualization helpers
  scripts/
    build_memory_bank.py       # Build + save a category's memory bank (auto-manages .gitignore exceptions)
    finalize_category.py        # Compute AUROC/IoU/threshold for a category, write to category_config.json
    visualize_heatmap.py          # One-example-per-defect-type heatmap visualization
    ...                              # Every other pipeline step as a runnable, documented entry point
  app/streamlit_app.py              # Interactive demo: category select, upload -> score + heatmap
  outputs/
    category_config.json            # Single source of truth: metrics + threshold per category
    memory_bank_<category>.pt         # Saved coresets (committed — needed for the deployed demo)
    heatmap_<category>.png              # Example localization visuals
  data/raw/                             # MVTec AD category folders (not tracked in git — see Setup)
```

## Setup and reproduction

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download `carpet`, `bottle`, and `wood` from the [official MVTec AD downloads page](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads), then extract each into `data/raw/`:

```bash
mv ~/Downloads/carpet.tar.xz ~/Downloads/bottle.tar.xz ~/Downloads/wood.tar.xz data/raw/
cd data/raw
tar -xf carpet.tar.xz
tar -xf bottle.tar.xz
tar -xf wood.tar.xz
cd ../..
```

You should now have `data/raw/carpet/`, `data/raw/bottle/`, and `data/raw/wood/`, each with `train/`, `test/`, and `ground_truth/` subfolders.

Then, for each category:

```bash
# build the memory bank (a few minutes each on Apple Silicon MPS)
python -m scripts.build_memory_bank --category carpet
python -m scripts.build_memory_bank --category bottle
python -m scripts.build_memory_bank --category wood

# compute AUROC / pixel-AUROC / IoU / threshold, written to outputs/category_config.json
python -m scripts.finalize_category --category carpet
python -m scripts.finalize_category --category bottle
python -m scripts.finalize_category --category wood
```

Finally, launch the dashboard — it reads `outputs/category_config.json` directly, so all three categories appear in the dropdown automatically, with no further setup:

```bash
streamlit run app/streamlit_app.py
```

All intermediate/diagnostic scripts (`check_*.py`, `sanity_check.py`, `visualize_heatmap.py`, `evaluate_detection.py`, `evaluate_localization.py`, etc.) are standalone, documented entry points under `scripts/` for inspecting each stage of the pipeline individually; they are not required for the steps above.

## Adding a new category

The pipeline is fully data-driven — adding a new MVTec AD category (or any dataset following the same `train/test/ground_truth` folder structure) requires no code changes, only running the existing scripts against the new category name. This was validated directly while building this project: **wood** was added this way, after carpet and bottle were already working.

1. **Download and extract** the category into `data/raw/<category>/`:
```bash
   mv ~/Downloads/<category>.tar.xz data/raw/
   cd data/raw && tar -xf <category>.tar.xz && cd ../..
```

2. **Build its memory bank** — this also automatically adds the required `.gitignore` exception for the new category's saved bank, so it isn't silently excluded from version control:
```bash
   python -m scripts.build_memory_bank --category <category>
```

3. **Compute its metrics and threshold** — appends a new entry to `outputs/category_config.json` alongside any existing categories, without overwriting them:
```bash
   python -m scripts.finalize_category --category <category>
```

4. **Run the dashboard** — the new category appears in the dropdown automatically, since the app reads its list of categories directly from `category_config.json`:
```bash
   streamlit run app/streamlit_app.py
```

**One manual judgment call, by design:** `build_memory_bank.py`'s `--target-size` (default 2200) controls the coreset size and may be worth tuning per category — a texture category and a highly complex object category don't necessarily warrant the same reference-bank size. If a new category's AUROC comes out lower than expected, this is the first parameter to revisit rather than something to auto-tune blindly.

## Limitations

- **Fixed photographic conditions.** Like any real inspection-camera setup, this system is specialized to consistent lighting, framing, and background matching its training images — confirmed directly by testing a web-sourced carpet photo, which was flagged anomalous despite having no real defect, simply due to different lighting/camera/fiber characteristics than MVTec AD's own photos. This is by design, matching how real factory inspection stations work (one fixed camera per product line), not a shortcoming unique to this implementation.
- **Localization resolution is coarse.** The 28×28 patch grid, upsampled to image resolution, cannot produce pixel-precise boundaries — defect *location* is reliable, but boundary shape is approximate, which is reflected in the IoU numbers above.
- **Threshold calibration requires some labeled defective examples.** The reported thresholds were validated against MVTec AD's labeled test set. A genuinely new deployment with zero defective examples available would need to start with a conservative, normal-data-only threshold and refine it from live production feedback over time.
- **Three of fifteen MVTec AD categories evaluated**, chosen to represent texture and object defect types; the pipeline generalizes to the remaining categories via the same steps (see "Adding a new category"), but their specific results are not yet measured.
- Dataset is licensed CC BY-NC-SA 4.0 — non-commercial use only.
