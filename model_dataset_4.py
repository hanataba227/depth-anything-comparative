import os
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class DepthDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.rgb_paths = []
        self.depth_paths = []
        self.transform = transform
        self.depth_transform = transforms.ToTensor()

        rgb_root = os.path.join(root_dir, "rgb")
        depth_root = os.path.join(root_dir, "pseudo_depth")

        for scene in os.listdir(rgb_root):
            rgb_scene_path = os.path.join(rgb_root, scene)
            depth_scene_path = os.path.join(depth_root, scene)

            for fname in os.listdir(rgb_scene_path):
                if not fname.endswith(('.png', '.jpg', '.jpeg')):
                    continue

                self.rgb_paths.append(os.path.join(rgb_scene_path, fname))
                self.depth_paths.append(os.path.join(depth_scene_path, os.path.splitext(fname)[0] + ".png"))

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb = Image.open(self.rgb_paths[idx]).convert("RGB")
        depth = cv2.imread(self.depth_paths[idx], cv2.IMREAD_GRAYSCALE) / 255.0  # [0, 1] 정규화
        depth = torch.from_numpy(depth).unsqueeze(0).float()

        if self.transform:
            rgb = self.transform(rgb)

        return rgb, depth
