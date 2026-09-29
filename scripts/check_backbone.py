import torch

from src.data.dataset import MVTecDataset
from src.models.backbone import PatchFeatureExtractor

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dataset = MVTecDataset("data/raw", "carpet", "train")
    extractor = PatchFeatureExtractor(device=device)

    img = dataset[0]["image"].unsqueeze(0)  # add batch dimension
    feat = extractor(img)

    print("device used:", device)
    print("feature shape:", feat.shape)

if __name__ == "__main__":
    main()