# evaluate.py
import os
import sys
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from dataset.ddad_dataset import DdadDataset
from depth_anything_v2.dpt import DepthAnythingV2

def compute_metrics(pred, target):
    thresh = torch.max((target / pred), (pred / target))
    d1 = (thresh < 1.25).float().mean()
    d2 = (thresh < 1.25 ** 2).float().mean()
    d3 = (thresh < 1.25 ** 3).float().mean()
    mae = torch.abs(pred - target).mean()
    rmse = torch.sqrt(((pred - target) ** 2).mean())
    return mae.item(), rmse.item(), d1.item(), d2.item(), d3.item()

def main():
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
    val_dataset = torch.utils.data.Subset(dataset, range(500))
    loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = DepthAnythingV2(encoder="vitb")
    model.load_state_dict(torch.load("checkpoints/epoch01.pth"))
    model.to(device)
    model.eval()

    metrics = []
    with torch.no_grad():
        for img, depth in tqdm(loader, desc="Evaluating"):
            img, depth = img.to(device), depth.to(device)
            pred = model(img)
            depth = F.interpolate(depth, size=pred.shape[-2:], mode="bilinear", align_corners=False)
            pred = torch.clamp(pred, min=1e-6)
            depth = torch.clamp(depth, min=1e-6)
            metrics.append(compute_metrics(pred, depth))

    mae, rmse, d1, d2, d3 = map(lambda x: sum(x)/len(x), zip(*metrics))
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, δ1: {d1:.4f}, δ2: {d2:.4f}, δ3: {d3:.4f}")

if __name__ == "__main__":
    main()
