from src.data.dataset import MVTecDataset


def main():
    dataset = MVTecDataset("data/raw", "carpet", "test")
    print("dataset size:", len(dataset))

    sample = dataset[0]
    print("image shape:", sample["image"].shape)
    print("mask shape:", sample["mask"].shape)
    print("label:", sample["label"])
    print("defect type:", sample["defect_type"])
    print("mask pixel count:", sample["mask"].sum().item())


if __name__ == "__main__":
    main()