import time

import torch

from src.data.dataset import MVTecDataset
from src.models.backbone import PatchFeatureExtractor
from src.models.memory_bank import collect_features, greedy_coreset


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dataset = MVTecDataset("data/raw", "carpet", "train")
    extractor = PatchFeatureExtractor(device=device)

    features = collect_features(extractor, dataset)
    print("pool shape:", features.shape)

    target_size = 500
    start = time.time()
    coreset, indices = greedy_coreset(features, target_size, device=device)
    elapsed = time.time() - start

    print("coreset shape:", coreset.shape)
    print("num unique indices:", len(set(indices.tolist())))
    print(f"time for {target_size} points: {elapsed:.1f}s")


if __name__ == "__main__":
    main()