import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import MVTecDataset
from src.models.backbone import PatchFeatureExtractor
from src.models.memory_bank import score_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="carpet")
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    memory_bank = torch.load(f"outputs/memory_bank_{args.category}.pt").to(device)
    dataset = MVTecDataset("data/raw", args.category, "test")
    extractor = PatchFeatureExtractor(device=device)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    scores, labels = [], []
    for batch in loader:
        imgs = batch["image"]
        feat = extractor(imgs)
        B, C, H, W = feat.shape
        patch_features = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        s, _ = score_batch(patch_features, memory_bank)
        scores.extend(s.cpu().tolist())
        labels.extend(batch["label"].tolist())

    scores = np.array(scores)
    labels = np.array(labels)
    preds = (scores >= args.threshold).astype(int)

    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    print(f"category: {args.category}, threshold: {args.threshold}")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"false positive rate (good images wrongly flagged): {fp / (fp + tn):.2%}")
    print(f"recall (defective images correctly caught): {tp / (tp + fn):.2%}")


if __name__ == "__main__":
    main()