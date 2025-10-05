import os
import torch
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
from tqdm import tqdm
import sys

sys.path.append("C:/project/Depth-Anything-V2")
from depth_anything_v2.dpt import DepthAnythingV2

# 설정
MODEL_TYPE = "vitb"
CHECKPOINT_PATH = "C:/project/depth-anything-comparative/checkpoints/depth_anything_v2_vitb.pth"
INPUT_ROOT = "dataset/rgb"
OUTPUT_ROOT = "dataset/pseudo_depth"

# ✅ 모델 설정 값 지정
model_config = {
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]}
}

# 모델 로드
model = DepthAnythingV2(**model_config[MODEL_TYPE])
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
model.eval().cuda()

# 전처리
transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

# 디렉토리 순회
for scene_name in tqdm(os.listdir(INPUT_ROOT), desc="전체 씬"):
    scene_rgb_path = os.path.join(INPUT_ROOT, scene_name)
    scene_out_path = os.path.join(OUTPUT_ROOT, scene_name)
    os.makedirs(scene_out_path, exist_ok=True)

    for file_name in tqdm(os.listdir(scene_rgb_path), desc=f"▶ {scene_name}", leave=False):
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(scene_rgb_path, file_name)
        image = Image.open(img_path).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).cuda()

        with torch.no_grad():
            pred = model(img_tensor)
            pred_depth = pred.squeeze().cpu().numpy()

        # Normalize
        depth_norm = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-8)
        depth_img = (depth_norm * 255).astype(np.uint8)

        # 저장 경로
        save_path = os.path.join(scene_out_path, os.path.splitext(file_name)[0] + ".png")
        cv2.imwrite(save_path, depth_img)
