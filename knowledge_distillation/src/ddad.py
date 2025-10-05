import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import json

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ], dtype=np.float32)

def create_depth_map(points_3d, img_size, K, max_depth=80.0):
    H, W = img_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    points_3d = torch.tensor(points_3d, device=device).float()
    depths = points_3d[:, 2].clone()

    uv = torch.matmul(points_3d, K.t())
    uv = uv[:, :2] / uv[:, 2:3]

    u = uv[:, 0]
    v = uv[:, 1]

    mask = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (depths > 0) & (depths < max_depth)

    print(f"Total points: {points_3d.shape[0]}")
    print(f"Points in image bounds: {mask.sum().item()}")

    u = u[mask].long()
    v = v[mask].long()
    depths = depths[mask]

    depth_map = torch.zeros((H, W), device=device)
    depth_map[v, u] = depths

    return depth_map.cpu()

class DDADDepthDataset(Dataset):
    def __init__(self, root_dir, cameras=['CAMERA_01'], img_size=(384, 384), max_depth=80.0):
        self.root_dir = root_dir
        self.cameras = cameras
        self.img_size = img_size
        self.max_depth = max_depth
        self.img_transform = T.Compose([
            T.Resize(self.img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        self.samples = self._parse_dataset()

    def _parse_dataset(self):
        samples = []
        scene_dirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        print(f"Scene directories found: {scene_dirs}")

        for scene in scene_dirs:
            scene_path = os.path.join(self.root_dir, scene)
            calib_dir = os.path.join(scene_path, 'calibration')
            if not os.path.exists(calib_dir):
                continue
            calib_files = [f for f in os.listdir(calib_dir) if f.endswith('.json')]
            if not calib_files:
                continue
            calib_path = os.path.join(calib_dir, calib_files[0])

            for camera_name in self.cameras:
                rgb_dir = os.path.join(scene_path, 'rgb', camera_name)
                pc_dir = os.path.join(scene_path, 'point_cloud', 'LIDAR')
                if not os.path.exists(rgb_dir) or not os.path.exists(pc_dir):
                    continue

                rgb_files = sorted(os.listdir(rgb_dir))
                pc_files = sorted(os.listdir(pc_dir))

                for img_file in rgb_files:
                    img_id_str = img_file.split('.')[0]
                    try:
                        img_id = int(img_id_str)
                    except ValueError:
                        print(f"Skipping non-numeric filename: {img_file}")
                        continue

                    pc_ids = [int(f.split('.')[0]) for f in pc_files]
                    best_pc_idx = np.argmin([abs(pc_id - img_id) for pc_id in pc_ids])
                    best_match = pc_files[best_pc_idx]

                    img_path = os.path.join(rgb_dir, img_file)
                    pc_path = os.path.join(pc_dir, best_match)
                    samples.append((img_path, pc_path, calib_path, camera_name))

        print(f"Total samples collected: {len(samples)}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, pc_path, calib_path, camera_name = self.samples[idx]

        img = Image.open(img_path).convert('RGB')
        img = self.img_transform(img)

        pc_data = np.load(pc_path)
        points = pc_data['data'][:, :3] 
        print(f"Point cloud loaded: shape={points.shape}, min={points.min(axis=0)}, max={points.max(axis=0)}")

        with open(calib_path, 'r') as f:
            calib = json.load(f)

        names = calib['names']
        intrinsics_list = calib['intrinsics']
        extrinsics_list = calib['extrinsics']

        idx_cam = names.index(camera_name)
        intrinsic_data = intrinsics_list[idx_cam]
        extrinsic_data = extrinsics_list[idx_cam]

        K = np.array([
            [intrinsic_data['fx'], intrinsic_data['skew'], intrinsic_data['cx']],
            [0, intrinsic_data['fy'], intrinsic_data['cy']],
            [0, 0, 1]
        ], dtype=np.float32)

        rot = extrinsic_data['rotation']
        trans = extrinsic_data['translation']

        qw, qx, qy, qz = rot['qw'], rot['qx'], rot['qy'], rot['qz']
        rotation_matrix = quaternion_to_rotation_matrix(qw, qx, qy, qz)

        translation_vector = np.array([trans['x'], trans['y'], trans['z']], dtype=np.float32)

        extrinsic_matrix = np.eye(4, dtype=np.float32)
        extrinsic_matrix[:3, :3] = rotation_matrix
        extrinsic_matrix[:3, 3] = translation_vector

        extrinsic_matrix = np.linalg.inv(extrinsic_matrix)

        points_hom = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        points_cam = (extrinsic_matrix @ points_hom.T).T[:, :3]

        points_cam[:, 1] *= -1
        points_cam[:, 2] *= -1

        print("Extrinsic Matrix:\n", extrinsic_matrix)
        print("Sample points after transform (first 5):\n", points_cam[:5])
        print("Min Z:", points_cam[:, 2].min(), "Max Z:", points_cam[:, 2].max())

        depth_map = create_depth_map(points_cam, self.img_size, torch.tensor(K), self.max_depth)

        return img, depth_map
