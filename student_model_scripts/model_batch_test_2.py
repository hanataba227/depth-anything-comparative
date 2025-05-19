import os
import torch
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
from tqdm import tqdm
import sys

# 상위 경로에서 dpt 접근 가능하도록 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from depth_anything_v2.dpt import DepthAnythingV2

MAX_IMAGES = 1200  # 엔트리 이미지수 제한

# 설정
MODEL_TYPE = "vitb"
CHECKPOINT_PATH = "checkpoints/depth_anything_v2_vitb.pth"
INPUT_FOLDER = "dataset/images"
OUTPUT_FOLDER = "dataset/pseudo_depth"

# ✅ 모델 설정 값
model_config = {
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]}
}

# 디바이스 설정
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print(f"✅ Using device: {DEVICE}")

# 모델 로드
model = DepthAnythingV2(**model_config[MODEL_TYPE])
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model = model.to(DEVICE).eval()

# 전처리
transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

# 출력 폴더 생성
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 이미지 추론 
image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:MAX_IMAGES]
print(f"📁 총 이미지 수 (제한): {len(image_files)}")

for fname in tqdm(image_files, desc="🔍 추론 중"):
    img_path = os.path.join(INPUT_FOLDER, fname)
    image = Image.open(img_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(img_tensor)
        pred_depth = pred.squeeze().cpu().numpy()

    # ✅ Shape 검증 (2D가 아닐 경우 건너뜀)
    if pred_depth.ndim != 2:
        print(f"⚠️ Skipped {fname} - Invalid depth shape: {pred_depth.shape}")
        continue

    # Normalize for visualization
    depth_norm = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-8)
    depth_img = (depth_norm * 255).astype(np.uint8)

    base_name = os.path.splitext(fname)[0]
    png_path = os.path.join(OUTPUT_FOLDER, base_name + ".png")
    npy_path = os.path.join(OUTPUT_FOLDER, base_name + "_depth.npy")

    cv2.imwrite(png_path, depth_img)
    np.save(npy_path, pred_depth)

print("🎉 모든 이미지 추론 및 저장 완료!")