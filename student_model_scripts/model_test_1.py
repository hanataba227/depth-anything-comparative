import torch
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
import sys
import os

# 경로 추가 (상위 폴더에 있는 dpt.py 접근용)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from depth_anything_v2.dpt import DepthAnythingV2

# ✅ 경로 설정 - 현재 디렉토리 기준
CHECKPOINT_PATH = "checkpoints/depth_anything_v2_vitb.pth"
IMAGE_PATH = "sample.jpg"
OUTPUT_PATH = "depth_output.png"

# ✅ 디바이스 설정
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print(f"✅ Using device: {DEVICE}")

# ✅ 모델 로드 (vitb에 맞춰서 설정)
model_configs = {
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]}
}
model = DepthAnythingV2(**model_configs['vitb'])
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model = model.to(DEVICE).eval()

# ✅ 이미지 전처리
image = Image.open(IMAGE_PATH).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])
img_tensor = transform(image).unsqueeze(0).to(DEVICE)

# ✅ 추론
with torch.no_grad():
    pred = model(img_tensor)
    pred_depth = pred.squeeze().cpu().numpy()

# ✅ 결과 저장
depth_norm = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-8)
depth_img = (depth_norm * 255).astype(np.uint8)
cv2.imwrite(OUTPUT_PATH, depth_img)

print(f"🎉 저장 완료: {OUTPUT_PATH}")