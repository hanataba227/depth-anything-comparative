import torch
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
import sys
sys.path.append("C:/project/Depth-Anything-V2")
from depth_anything_v2.dpt import DepthAnythingV2 as DepthAnything

# 모델 로드
model = DepthAnything("vitb")
model.load_state_dict(torch.load("checkpoints/depth_anything_vitb14.pth"))
model.eval().cuda()

# 이미지 전처리
image_path = "sample.jpg"  # 테스트용 이미지
image = Image.open(image_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])
img_tensor = transform(image).unsqueeze(0).cuda()

# 추론
with torch.no_grad():
    pred = model(img_tensor)
    pred_depth = pred.squeeze().cpu().numpy()

# 저장
depth_norm = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min())
cv2.imwrite("depth_output.png", (depth_norm * 255).astype(np.uint8))
