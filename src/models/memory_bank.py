import torch
from torch.utils.data import DataLoader
import torch.nn .functional as F

@torch.no_grad()
def collect_features(extractor, dataset, batch_size=8):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_features = []

    for batch in loader:
        imgs = batch["image"]
        feat = extractor(imgs) # shape: (B, C, H, W)
        B, C, H, W = feat.shape
        # move channels last, then flatten spatial grid into individual path vectors
        feat = feat.permute(0, 2, 3, 1).reshape(B * H * W, C)
        all_features.append(feat.cpu())

    return torch.cat(all_features, dim=0) # shape: (N_total_patches, C)

def greedy_coreset(features, target_size, seed=0, device="cpu"):
    features = features.to(device)
    N = features.shape[0]
    generator = torch.Generator().manual_seed(seed)

    selected_indices = []
    first_idx = torch.randint(0, N, (1,), generator=generator).item()
    selected_indices.append(first_idx)

    # distance from every point to the nearest currenlty selected point
    min_dists = torch.cdist(features, features[first_idx].unsqueeze(0)).squeeze(1) # shape: (N,)

    for _ in range(target_size - 1):
        next_idx = torch.argmax(min_dists).item()
        selected_indices.append(next_idx)

        # update min_dists
        new_dists = torch.cdist(features, features[next_idx].unsqueeze(0)).squeeze(1)
        min_dists = torch.minimum(min_dists, new_dists)

    return features[selected_indices], torch.tensor(selected_indices)

def score_batch(patch_features, memory_bank):
    # patch_features: (B, num_patches, C)
    # memory_bank: (M, C)
    B, num_patches, C = patch_features.shape
    flat = patch_features.reshape(B * num_patches, C)

    dists = torch.cdist(flat, memory_bank) # (B * num_patches, M)
    min_dists, _ = dists.min(dim=1) # nearest bank neighbour per patch

    min_dists = min_dists.reshape(B, num_patches) # (B, num_patches)
    image_scores = min_dists.max(dim=1).values # (B, )

    return image_scores, min_dists

def make_heatmap(patch_min_dists, grid_size, image_size):
    # patch_min_dists: (num_patches,) for ONE IMAGE
    heatmap = patch_min_dists.reshape(1, 1, grid_size, grid_size)
    heatmap = F.interpolate(heatmap, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return heatmap.squeeze() # (image_size, image_size)