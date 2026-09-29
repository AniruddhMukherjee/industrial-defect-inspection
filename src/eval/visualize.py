import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD


def denormalize(img):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (img * std + mean).clamp(0, 1)


def show_samples(dataset, indices, save_path):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, axes = plt.subplots(len(indices), 3, figsize=(9, 3 * len(indices)))
    for row, idx in enumerate(indices):
        s = dataset[idx]
        img = denormalize(s["image"]).permute(1, 2, 0).numpy()
        mask = s["mask"][0].numpy()
        overlay = np.ma.masked_where(mask == 0, mask)

        axes[row, 0].imshow(img)
        axes[row, 0].set_title(s["defect_type"])
        axes[row, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("mask")
        axes[row, 2].imshow(img)
        axes[row, 2].imshow(overlay, cmap="autumn", alpha=0.6)
        axes[row, 2].set_title("overlay")
        for ax in axes[row]:
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)