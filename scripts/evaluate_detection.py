import argparse

import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.data.dataset import MVTecDataset
from src.models.backbone import PatchFeatureExtractor
from src.models.memory_bank import score_batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="carpet")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    category = args.category

    memory_bank = torch.load(f"outputs/memory_bank_{category}.pt").to(device)
    dataset = MVTecDataset("data/raw", category, "test")
    extractor = PatchFeatureExtractor(device=device)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    all_scores = []
    all_labels = []

    for batch in loader:
        imgs = batch["image"]
        labels = batch["label"]

        feat = extractor(imgs) # shape: (B, C, H, W)
        B, C, H, W = feat.shape
        patch_features = feat.permute(0, 2, 3, 1).reshape(B, H * W, C) # shape: (B, num_patches, C)

        scores, _ = score_batch(patch_features, memory_bank)

        all_scores.extend(scores.cpu().tolist())
        all_labels.extend(labels.tolist())

    auroc = roc_auc_score(all_labels, all_scores)
    print(f"category: {category}")
    print(f"num test images: {len(all_labels)}")
    print(f"image-level AUROC: {auroc:.4f}")

if __name__ == "__main__":
    main()