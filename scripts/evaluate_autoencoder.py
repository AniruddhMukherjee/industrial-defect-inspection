import argparse

import torch
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.data.dataset import MVTecDataset
from src.models.autoencoder import ConvAutoencoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="carpet")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load(f"outputs/autoencoder_{args.category}.pt"))
    model.eval()

    dataset = MVTecDataset("data/raw", args.category, "test", normalize=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            labels = batch["label"]

            recon = model(imgs)
            error_map = ((imgs - recon) ** 2).mean(dim=1)  # (B, H, W), avg over color channels

            for i in range(error_map.shape[0]):
                blurred = gaussian_filter(error_map[i].cpu().numpy(), sigma=4)
                all_scores.append(blurred.max())  # image score = worst smoothed region

            all_labels.extend(labels.tolist())

    auroc = roc_auc_score(all_labels, all_scores)
    print(f"category: {args.category}")
    print(f"autoencoder image-level AUROC: {auroc:.4f}")


if __name__ == "__main__":
    main()