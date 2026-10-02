import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.data.dataset import MVTecDataset
from src.eval.visualize import denormalize
from src.models.backbone import PatchFeatureExtractor
from src.models.memory_bank import score_batch, make_heatmap

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="carpet")
    parser.add_argument("--num-samples", type=int, default=4)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    memory_bank = torch.load(f"outputs/memory_bank_{args.category}.pt").to(device)
    dataset = MVTecDataset("data/raw", args.category, "test")
    extractor = PatchFeatureExtractor(device=device)

    # grab the first few DEFECTIVE samples (skip "good" ones, nothing to localize there) -- might fail idk
    seen_types = {}
    for i, s in enumerate(dataset.samples):
        if s[2] == 1:
            seen_types.setdefault(s[3], i)
    defective_indices = list(seen_types.values())[: args.num_samples]

    fig, axes = plt.subplots(len(defective_indices), 3, figsize=(9,3 * len(defective_indices)))

    for row, idx in enumerate(defective_indices):
        sample = dataset[idx]
        img_tensor = sample["image"].unsqueeze(0)

        feat = extractor(img_tensor) # (1, C, H, W)
        B, C, H, W = feat.shape
        patch_features = feat.permute(0, 2, 3, 1).reshape(B, H * W, C) # (1, num_patches, C)

        _, min_dists = score_batch(patch_features, memory_bank)
        heatmap = make_heatmap(min_dists[0], grid_size=H, image_size=224).cpu()

        img = denormalize(sample["image"]).permute(1, 2, 0).numpy()
        gt_mask = sample["mask"][0].numpy()

        axes[row, 0].imshow(img)
        axes[row, 0].set_title(sample["defect_type"])
        axes[row, 1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("ground truth")
        axes[row, 2].imshow(img)
        axes[row, 2].imshow(heatmap, cmap="jet", alpha=0.5)
        axes[row, 2].set_title("predicted_heatmap")
        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    save_path = f"outputs/heatmap_{args.category}.png"
    plt.savefig(save_path, dpi=100)
    print("saved", save_path)

if __name__ == "__main__":
    main()