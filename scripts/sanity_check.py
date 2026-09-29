import argparse

from src.data.dataset import MVTecDataset
from src.eval.visualize import show_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="carpet")
    args = parser.parse_args()

    dataset = MVTecDataset("data/raw", args.category, "test")

    first_of_each = {}
    for i, sample in enumerate(dataset.samples):
        defect_type = sample[3]
        first_of_each.setdefault(defect_type, i)

    save_path = f"outputs/sanity_{args.category}.png"
    show_samples(dataset, list(first_of_each.values()), save_path)
    print("saved", save_path, "for defect types:", list(first_of_each.keys()))


if __name__ == "__main__":
    main()