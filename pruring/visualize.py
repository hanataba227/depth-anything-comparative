# visualize.py
import os
import sys
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn.functional as F
from random import sample

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from dataset.ddad_dataset import DdadDataset
from depth_anything_v2.dpt import DepthAnythingV2

def visualize(pred, gt, rgb, idx):
    pred = pred.squeeze().cpu().numpy()
    gt = gt.squeeze().cpu().numpy()
    rgb = rgb.squeeze().permute(1, 2, 0).cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(rgb)
    axs[0].set_title("RGB")
    axs[1].imshow(gt, cmap="inferno")
    axs[1].set_title("GT Depth")
    axs[2].imshow(pred, cmap="inferno")
    axs[2].set_title("Pred Depth")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"visuals/sample_{idx:03d}.png")
    plt.close()

def main():
    os.makedirs("visuals", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    dataset = DdadDataset(
        root_dir="C:/project/depth-anything-comparative_2",
        transform=transform,
        camera="CAMERA_01"
    )

    # 랜덤 10개 샘플링
    indices = sample(range(len(dataset)), 10)
    loader = DataLoader(Subset(dataset, indices), batch_size=1)

    model = DepthAnythingV2(encoder="vitb")
    model.load_state_dict(torch.load("checkpoints/epoch09.pth"))
    model.to(device)
    model.eval()

    with torch.no_grad():
        for idx, (img, gt) in enumerate(tqdm(loader, desc="Visualizing")):
            img, gt = img.to(device), gt.to(device)
            pred = model(img)
            gt = F.interpolate(gt, size=pred.shape[-2:], mode="bilinear", align_corners=False)
            visualize(pred, gt, img, idx)

if __name__ == "__main__":
    main()
