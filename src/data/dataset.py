from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class MVTecDataset(Dataset):
    def __init__(self, root, category, split="train", resize = 256, crop=224, normalize=True):
        assert split in ("train", "test")
        self.root = Path(root) / category
        self.split = split
        self.resize = resize
        self.crop = crop
        self.normalize = normalize
        self.samples = self._collect()

    def _collect(self):
        # each sample: (image_path, mask_path or None, label, defect_type)
        samples = []
        for type_dir in sorted((self.root/ self.split).iterdir()):
            if not type_dir.is_dir():
                continue
            defect = type_dir.name
            for img_path in sorted(type_dir.glob("*.png")):
                if defect == "good":
                    samples.append((img_path, None, 0, defect))
                else:
                    mask_path = self.root / "ground_truth" / defect / f"{img_path.stem}_mask.png"
                    samples.append((img_path, mask_path, 1, defect))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label, defect = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        img = TF.resize(img, self.resize)
        img = TF.center_crop(img, self.crop)
        img = TF.to_tensor(img)
        if self.normalize:
            img = TF.normalize(img, IMAGENET_MEAN, IMAGENET_STD)

        if mask_path is None:
            mask = torch.zeros(1, self.crop, self.crop)
        else:
            m = Image.open(mask_path).convert("L")
            m = TF.resize(m, self.resize, interpolation=TF.InterpolationMode.NEAREST)
            m = TF.center_crop(m, self.crop)
            mask = (TF.to_tensor(m) > 0.5).float()

        return {
            "image": img,
            "mask": mask,
            "label": label,
            "defect_type": defect,
            "path": str(img_path),
        }