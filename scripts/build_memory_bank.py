import time

import torch

from src.data.dataset import MVTecDataset
from src.models.backbone import PatchFeatureExtractor
from src.models.memory_bank import collect_features, greedy_coreset

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    category = "carpet"
    target_size = 2200

    dataset = MVTecDataset("data/raw", category, "train")
    extractor = PatchFeatureExtractor(device=device)

    print("collecting features...")
    features = collect_features(extractor, dataset)
    print("pool shape:", features.shape)

    print(f"running coreset selection for {target_size} points...")
    start = time.time()
    coreset, _ = greedy_coreset(features, target_size, device=device)
    elapsed = time.time() - start
    print(f"done in {elapsed:.1f}s, coreset shape: {coreset.shape}")

    save_path = f"outputs/memory_bank_{category}.pt"
    torch.save(coreset, save_path)
    print("saved to", save_path)

if __name__ == "__main__":
    main()