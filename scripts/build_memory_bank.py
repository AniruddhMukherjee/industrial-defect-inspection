import argparse
import time
from pathlib import Path

import torch

from src.data.dataset import MVTecDataset
from src.models.backbone import PatchFeatureExtractor
from src.models.memory_bank import collect_features, greedy_coreset


def ensure_gitignore_exception(category):
    gitignore_path = Path(".gitignore")
    exception_line = f"!outputs/memory_bank_{category}.pt"

    content = gitignore_path.read_text() if gitignore_path.exists() else ""
    if exception_line in content:
        return  # already there, nothing to do

    if not content.endswith("\n"):
        content += "\n"
    content += exception_line + "\n"
    gitignore_path.write_text(content)
    print(f"added gitignore exception for {category}'s memory bank")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="carpet")
    parser.add_argument("--target-size", type=int, default=2200)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    dataset = MVTecDataset("data/raw", args.category, "train")
    extractor = PatchFeatureExtractor(device=device)

    print("collecting features...")
    features = collect_features(extractor, dataset)
    print("pool shape:", features.shape)

    print(f"running coreset selection for {args.target_size} points...")
    start = time.time()
    coreset, _ = greedy_coreset(features, args.target_size, device=device)
    elapsed = time.time() - start
    print(f"done in {elapsed:.1f}s, coreset shape: {coreset.shape}")

    save_path = f"outputs/memory_bank_{args.category}.pt"
    torch.save(coreset, save_path)
    print("saved to", save_path)

    ensure_gitignore_exception(args.category)


if __name__ == "__main__":
    main()