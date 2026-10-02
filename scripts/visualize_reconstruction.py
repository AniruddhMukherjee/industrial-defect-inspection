import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.data.dataset import MVTecDataset
from src.models.autoencoder import ConvAutoencoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="carpet")
    parser.add_argument("--num-samples", type=int, default=4)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load(f"outputs/autoencoder_{args.category}.pt"))
    model.eval()

    dataset = MVTecDataset("data/raw", args.category, "test", normalize=False)
    defective_indices = [i for i, s in enumerate(dataset.samples) if s[2] == 1][: args.num_samples]

    fig, axes = plt.subplots(len(defective_indices), 3, figsize=(9, 3 * len(defective_indices)))

    with torch.no_grad():
        for row, idx in enumerate(defective_indices):
            sample = dataset[idx]
            img = sample["image"].unsqueeze(0).to(device)
            recon = model(img)
            error = (img - recon).abs().squeeze(0).mean(dim=0).cpu()

            axes[row, 0].imshow(img.squeeze(0).permute(1, 2, 0).cpu().numpy())
            axes[row, 0].set_title(sample["defect_type"])
            axes[row, 1].imshow(recon.squeeze(0).permute(1, 2, 0).cpu().numpy())
            axes[row, 1].set_title("reconstruction")
            axes[row, 2].imshow(error, cmap="hot")
            axes[row, 2].set_title("abs error")
            for ax in axes[row]:
                ax.axis("off")

    plt.tight_layout()
    save_path = f"outputs/reconstruction_{args.category}.png"
    plt.savefig(save_path, dpi=100)
    print("saved", save_path)


if __name__ == "__main__":
    main()