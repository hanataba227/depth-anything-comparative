import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class DepthEstimationMobileNetV2(nn.Module):
    def __init__(self, pretrained=True):
        super(DepthEstimationMobileNetV2, self).__init__()

        mobilenet = models.mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None).features

        # Encoder: Early to mid-stage layer 사용
        self.encoder = mobilenet[:14]

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # H/8

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # H/4
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # H/2

            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
