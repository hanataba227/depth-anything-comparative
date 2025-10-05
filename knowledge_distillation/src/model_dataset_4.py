import cv2
import torch
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class JsonDepthDataset(Dataset):
    def __init__(self, json_path, transform=None, resize=(224, 224)):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.transform = transform
        self.resize = resize

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        rgb_path = item['rgb'].replace("\\", "/")
        depth_path = item['depth'].replace("\\", "/")

        # RGB 처리
        rgb = Image.open(rgb_path).convert("RGB")
        if self.transform:
            rgb = self.transform(rgb)

        # Depth 처리
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        if depth_raw is None:
            raise FileNotFoundError(f"Cannot read depth: {depth_path}")

        depth_resized = cv2.resize(depth_raw, self.resize, interpolation=cv2.INTER_NEAREST)
        depth = torch.tensor(depth_resized / 255.0).unsqueeze(0).float()

        return rgb, depth