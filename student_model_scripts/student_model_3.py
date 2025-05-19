import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class DepthEstimationMobileNetV2(nn.Module):
    # ⚠️ Training recommendation: use F.smooth_l1_loss instead of F.l1_loss for better robustness.
    def __init__(self, pretrained=True):
        super(DepthEstimationMobileNetV2, self).__init__()

        mobilenet = models.mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None).features

        # Encoder
        self.encoder = mobilenet[:14]

        # Decoder (wider version)
        self.decoder = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x