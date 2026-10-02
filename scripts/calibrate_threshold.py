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
    parser.add_argument("--percentile", type=float, default=99.0)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    memory_bank = torch.load(f"outputs/memory_bank_{args.category}.pt").to(device)

    # training set = all normal images, exactly what a threshold should be calibrated against
    dataset = MVTecDataset("data/raw", args.category, "train")
    extractor = PatchFeatureExtractor(device=device)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    all_scores = []
    for batch in loader:
        imgs = batch["image"]
        feat = extractor(imgs)
        B, C, H, W = feat.shape
        patch_features = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        scores, _ = score_batch(patch_features, memory_bank)
        all_scores.extend(scores.cpu().tolist())

    scores = np.array(all_scores)
    threshold = np.percentile(scores, args.percentile)

    print(f"category: {args.category}")
    print(f"num normal training images scored: {len(scores)}")
    print(f"score distribution: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
    print(f"{args.percentile}th percentile threshold: {threshold:.4f}")


if __name__ == "__main__":
    main()