import argparse
from pathlib import Path

import torchvision.transforms.functional as TF
from PIL import Image

from src.data.dataset import MVTecDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument("--per-type", type=int, default=1)
    parser.add_argument("--good-count", type=int, default=2)
    args = parser.parse_args()

    dataset = MVTecDataset("data/raw", args.category, "test")
    out_dir = Path("samples") / args.category
    out_dir.mkdir(parents=True, exist_ok=True)

    taken = {}
    for img_path, _, _, defect in dataset.samples:
        limit = args.good_count if defect == "good" else args.per_type
        if taken.get(defect, 0) >= limit:
            continue
        taken[defect] = taken.get(defect, 0) + 1

        img = Image.open(img_path).convert("RGB")
        img = TF.resize(img, 256)
        img.save(out_dir / f"{defect}_{img_path.stem}.png")

    print(f"saved {sum(taken.values())} samples to {out_dir}")
    for defect, n in taken.items():
        print(f"  {defect}: {n}")


if __name__ == "__main__":
    main()