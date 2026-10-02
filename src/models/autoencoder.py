import math

import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_channels=256, target_mean=0.35):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.GroupNorm(8, 128), nn.ReLU(inplace=True),
            nn.Conv2d(128, latent_channels, 4, stride=2, padding=1), nn.GroupNorm(8, latent_channels), nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, 4, stride=2, padding=1), nn.GroupNorm(8, 128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid(),
        )

        self._init_output_bias(target_mean)

    def _init_output_bias(self, target_mean):
        # logit = inverse of sigmoid: start the final layer near the real data mean,
        # instead of a random point the optimizer has to crawl away from.
        logit = math.log(target_mean / (1 - target_mean))
        final_layer = self.decoder[-2]  # the last ConvTranspose2d, before Sigmoid
        nn.init.zeros_(final_layer.weight)
        nn.init.constant_(final_layer.bias, logit)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)