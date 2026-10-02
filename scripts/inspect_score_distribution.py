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

    good = scores[labels == 0]
    defective = scores[labels == 1]

    print(f"category: {args.category}")
    print(f"GOOD      (n={len(good)}): min={good.min():.4f} max={good.max():.4f} mean={good.mean():.4f}")
    print(f"DEFECTIVE (n={len(defective)}): min={defective.min():.4f} max={defective.max():.4f} mean={defective.mean():.4f}")
    print(f"gap: good.max()={good.max():.4f} vs defective.min()={defective.min():.4f}")


if __name__ == "__main__":
    main()