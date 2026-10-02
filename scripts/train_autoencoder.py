import argparse
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import MVTecDataset
from src.models.autoencoder import ConvAutoencoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="carpet")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    dataset = MVTecDataset("data/raw", args.category, "train", normalize=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = ConvAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_state = None

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in loader:
            imgs = batch["image"].to(device)

            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, imgs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)

        avg_loss = total_loss / len(dataset)
        marker = ""
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = copy.deepcopy(model.state_dict())
            marker = "  <- best so far"
        print(f"epoch {epoch + 1}/{args.epochs} - avg loss: {avg_loss:.6f}{marker}")

    save_path = f"outputs/autoencoder_{args.category}.pt"
    torch.save(best_state, save_path)
    print(f"saved BEST checkpoint (loss={best_loss:.6f}) to", save_path)


if __name__ == "__main__":
    main()