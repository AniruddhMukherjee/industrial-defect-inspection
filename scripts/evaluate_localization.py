import argparse

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve
from torch.utils.data import DataLoader

from src.data.dataset import MVTecDataset
from src.models.backbone import PatchFeatureExtractor
from src.models.memory_bank import score_batch, make_heatmap

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="carpet")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    memory_bank = torch.load(f"outputs/memory_bank_{args.category}.pt").to(device)
    dataset = MVTecDataset("data/raw", args.category, "test")
    extractor = PatchFeatureExtractor(device=device)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    all_pixel_scores = []
    all_pixel_labels = []

    for batch in loader:
        imgs = batch["image"]
        masks = batch["mask"] # shape: (B, 1, 244, 244)

        feat = extractor(imgs)
        B, C, H, W = feat.shape
        patch_features = feat.permute(0, 2, 3, 1). reshape(B, H * W, C)
        _, min_dists = score_batch(patch_features, memory_bank)

        for i in range(B):
            heatmap = make_heatmap(min_dists[i], grid_size=H, image_size=224).cpu()
            all_pixel_scores.append(heatmap.flatten().numpy())
            all_pixel_labels.append(masks[i, 0].flatten().numpy())

    scores = np.concatenate(all_pixel_scores)
    labels = np.concatenate(all_pixel_labels)

    pixel_auroc = roc_auc_score(labels, scores)

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]

    preds = (scores >= best_threshold).astype(int)
    intersection = np.logical_and(preds, labels).sum()
    union = np.logical_or(preds, labels).sum()
    best_iou = intersection / union if union > 0 else 0.0

    print(f"category: {args.category}")
    print(f"total pixels evaluated: {len(labels)}")
    print(f"defective pixel fraction: {labels.mean():.4f}")
    print(f"pixel-level AUROC: {pixel_auroc:.4f}")
    print(f"best achieveable IoU: {best_iou:.4f}")

if __name__ == "__main__":
    main()