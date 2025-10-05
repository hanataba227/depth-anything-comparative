# train.py
# Preprocessed DDAD (RGB + sparse depth) 학습용 전체 스크립트

import os
import sys
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from dataset.ddad_dataset import DdadDataset
from depth_anything_v2.dpt import DepthAnythingV2  # 수정된 import
from utils.loss import combined_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    full_dataset = DdadDataset(
        root_dir="C:/project/depth-anything-comparative_2",
        transform=transform,
        camera="CAMERA_01"
    )

    # 일부 샘플만 사용 (학습 속도 확인용)
    subset_size = min(500, len(full_dataset))
    train_dataset = Subset(full_dataset, range(subset_size))
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

    model = DepthAnythingV2(encoder="vitb")  # 기본 구조 그대로 사용
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    epochs = 20

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, depths in progress:
            depths = torch.clamp(depths, min=0.0, max=80.0)
            images, depths = images.to(device), depths.to(device)
            preds = model(images)

            if preds.ndim == 3:
                preds = preds.unsqueeze(1)

            if preds.ndim == 4 and depths.ndim == 4 and preds.shape[1] == depths.shape[1]:
                if preds.shape[-2:] != depths.shape[-2:]:
                    depths = F.interpolate(depths, size=preds.shape[-2:], mode="bilinear", align_corners=False)

            
            loss = combined_loss(preds, depths)

            if torch.isnan(loss):
                print("❌ Loss is NaN!")
                print("preds:", preds.min().item(), preds.max().item())
                print("depths:", depths.min().item(), depths.max().item())
                raise ValueError("Loss became NaN. Stopping training.")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Average Loss: {avg_loss:.4f}")

        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/epoch{epoch+1:02d}.pth")


if __name__ == "__main__":
    main()
