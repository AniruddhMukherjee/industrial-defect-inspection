import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

class PatchFeatureExtractor(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
        backbone = wide_resnet50_2(weights=weights)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)

        self.backbone = backbone.to(device)
        self.device = device
        self._features = {}

        self.backbone.layer2.register_forward_hook(self._make_hook("layer2"))
        self.backbone.layer3.register_forward_hook(self._make_hook("layer3"))

    def _make_hook(self, name):
        def hook(module, input, output):
            self._features[name] = output
        return hook

    @torch.no_grad()
    def forward(self, x):
        x = x.to(self.device)
        self._features = {}
        self.backbone(x) # runs the full network; hooks capture layer2/layer3 outputs

        f2 = self._features["layer2"]  # shape: (B, C2, H2, W2)
        f3 = self._features["layer3"]  # shape: (B, C3, H3, W3), H3 < H2

        f3_resized = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        combined = torch.cat([f2, f3_resized], dim=1)  # shape: (B, C2+C3, H2, W2)
        return combined
    