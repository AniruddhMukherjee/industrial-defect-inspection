import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve
from torch.utils.data import DataLoader

from src.data.dataset import MVTecDataset
from src.models.backbone import PatchFeatureExtractor
from src.models.memory_bank import make_heatmap, score_batch

CONFIG_PATH = Path("outputs/category_config.json")


def score_dataset(extractor, memory_bank, dataset, need_pixels=False):
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    scores, labels = [], []
    pixel_scores, pixel_labels = [], []

    for batch in loader:
        imgs = batch["image"]
        feat = extractor(imgs)
        B, C, H, W = feat.shape
        patch_features = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        s, min_dists = score_batch(patch_features, memory_bank)

        scores.extend(s.cpu().tolist())
        labels.extend(batch["label"].tolist())

        if need_pixels:
            masks = batch["mask"]
            for i in range(B):
                hm = make_heatmap(min_dists[i], grid_size=H, image_size=224).cpu()
                pixel_scores.append(hm.flatten().numpy())
                pixel_labels.append(masks[i, 0].flatten().numpy())

    result = {"scores": np.array(scores), "labels": np.array(labels)}
    if need_pixels:
        result["pixel_scores"] = np.concatenate(pixel_scores)
        result["pixel_labels"] = np.concatenate(pixel_labels)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    args = parser.parse_args()
    category = args.category

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    bank_path = f"outputs/memory_bank_{category}.pt"
    memory_bank = torch.load(bank_path, map_location="cpu").to(device)

    train_dataset = MVTecDataset("data/raw", category, "train")
    test_dataset = MVTecDataset("data/raw", category, "test")
    extractor = PatchFeatureExtractor(device=device)

    print(f"[{category}] scoring test set...")
    test_result = score_dataset(extractor, memory_bank, test_dataset, need_pixels=True)
    scores, labels = test_result["scores"], test_result["labels"]

    image_auroc = roc_auc_score(labels, scores)

    pixel_scores, pixel_labels = test_result["pixel_scores"], test_result["pixel_labels"]
    pixel_auroc = roc_auc_score(pixel_labels, pixel_scores)
    precision, recall, thresholds = precision_recall_curve(pixel_labels, pixel_scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1)
    pixel_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    preds = (pixel_scores >= pixel_threshold).astype(int)
    best_iou = np.logical_and(preds, pixel_labels).sum() / np.logical_or(preds, pixel_labels).sum()

    good = scores[labels == 0]
    defective = scores[labels == 1]
    threshold = (good.max() + defective.min()) / 2

    entry = {
        "num_train_images": len(train_dataset),
        "num_test_images": len(test_dataset),
        "memory_bank_path": bank_path,
        "image_auroc": round(float(image_auroc), 4),
        "pixel_auroc": round(float(pixel_auroc), 4),
        "best_iou": round(float(best_iou), 4),
        "good_score_max": round(float(good.max()), 4),
        "defective_score_min": round(float(defective.min()), 4),
        "threshold": round(float(threshold), 4),
    }

    config = {}
    if CONFIG_PATH.exists():
        config = json.loads(CONFIG_PATH.read_text())
    config[category] = entry
    CONFIG_PATH.write_text(json.dumps(config, indent=2))

    print(f"[{category}] done:")
    for k, v in entry.items():
        print(f"  {k}: {v}")
    print(f"saved to {CONFIG_PATH}")


if __name__ == "__main__":
    main()