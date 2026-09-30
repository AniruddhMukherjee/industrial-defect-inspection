import torch

from src.data.dataset import MVTecDataset
from src.models.backbone import PatchFeatureExtractor
from src.models.memory_bank import collect_features

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dataset = MVTecDataset("data/raw", "carpet", "train")
    extractor = PatchFeatureExtractor(device=device)

    features = collect_features(extractor, dataset)
    print("pool shape:", features.shape)

if __name__ == "__main__":
    main()