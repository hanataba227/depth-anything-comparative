import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class DepthDataset(Dataset):
    def __init__(self, ddad_root, etri_root, transform=None, max_ddad=5000, max_etri=1200):
        self.transform = transform
        
        # ✅ DDAD: vis_depth/ddad (png + npy 같이 존재 + shape 확인)
        self.ddad_dir = ddad_root
        self.ddad_files = []
        for fname in sorted(os.listdir(self.ddad_dir)):
            if not fname.endswith('.png'):
                continue
            npy_path = os.path.join(self.ddad_dir, fname.replace('.png', '_depth.npy'))
            if not os.path.exists(npy_path):
                continue
            try:
                depth = np.load(npy_path)
                if depth.ndim != 2:
                    continue
            except:
                continue
            self.ddad_files.append(fname)
        self.ddad_files = self.ddad_files[:max_ddad]

        # ✅ ETRI: dataset/images + dataset/pseudo_depth (depth가 있는 경우 + shape + 품질 필터링)
        self.etri_rgb_dir = os.path.join(etri_root, 'images')
        self.etri_depth_dir = os.path.join(etri_root, 'pseudo_depth')
        self.etri_files = []
        for fname in sorted(os.listdir(self.etri_rgb_dir)):
            if not (fname.endswith('.png') or fname.endswith('.jpg')):
                continue
            npy_path = os.path.join(
                self.etri_depth_dir,
                fname.replace('.png', '_depth.npy').replace('.jpg', '_depth.npy')
            )
            if not os.path.exists(npy_path):
                continue
            try:
                depth = np.load(npy_path)
                if depth.ndim != 2:
                    continue
                if depth.std() < 0.01:  # ✅ 품질 필터링
                    continue
            except:
                continue
            self.etri_files.append(fname)
        self.etri_files = self.etri_files[:max_etri]

        # ✅ 통합
        self.samples = [('ddad', f) for f in self.ddad_files] + [('etri', f) for f in self.etri_files]

        # ✅ 확인용 로그
        print(f"📊 Total training samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        source, fname = self.samples[idx]
        if source == 'ddad':
            rgb_path = os.path.join(self.ddad_dir, fname)
            depth_path = os.path.join(self.ddad_dir, fname.replace('.png', '_depth.npy'))
        else:
            rgb_path = os.path.join(self.etri_rgb_dir, fname)
            depth_path = os.path.join(self.etri_depth_dir, fname.replace('.png', '_depth.npy').replace('.jpg', '_depth.npy'))

        image = Image.open(rgb_path).convert("RGB")
        depth = np.load(depth_path).astype(np.float32)

        if self.transform:
            image = self.transform(image)

        depth = np.expand_dims(depth, axis=0)  # (1, H, W)
        return image, depth