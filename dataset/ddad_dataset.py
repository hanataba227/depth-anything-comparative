# dataset/ddad_dataset.py
# ddad.json 기반으로 DDAD 데이터셋을 로드

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class DdadDataset(Dataset):
    def __init__(self, root_dir, transform=None, camera="CAMERA_01"):
        self.root_dir = root_dir
        self.rgb_dir = os.path.join(root_dir, "output", "rgb", camera)
        self.depth_dir = os.path.join(root_dir, "output", "depth_gt_sparse", camera)
        self.transform = transform

        self.samples = []
        for filename in sorted(os.listdir(self.rgb_dir)):
            if filename.endswith(".png"):
                rgb_path = os.path.join(self.rgb_dir, filename)
                depth_path = os.path.join(self.depth_dir, filename)
                if os.path.exists(depth_path):
                    self.samples.append((rgb_path, depth_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, depth_path = self.samples[idx]
        image = Image.open(rgb_path).convert("RGB")
        depth = Image.open(depth_path)

        if self.transform:
            image = self.transform(image)
        depth = transforms.ToTensor()(depth).float()  # shape: (1, H, W)

        return image, depth