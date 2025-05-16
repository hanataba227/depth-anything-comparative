import os
import time
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from student_model_3 import DepthEstimationMobileNetV2

import sys
sys.path.append("C:/project/Depth-Anything-V2")
from depth_anything_v2.dpt import DepthAnythingV2

# === 설정 ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("📦 Using device:", DEVICE)

INPUT_SIZE = (224, 224)
UPSAMPLE_SIZE = (518, 518)
TEST_IMAGE_PATH = "test/sample.jpg"
NUM_TRIALS = 10

# === 전처리 ===
transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])
image = Image.open(TEST_IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(DEVICE)

# === 모델 로딩 ===
student_model = DepthEstimationMobileNetV2().to(DEVICE).eval()
student_model.load_state_dict(torch.load("trained_models/depth_student_mobilenetv2_1.pth"))

teacher_model = DepthAnythingV2(encoder="vitb", features=128, out_channels=[96, 192, 384, 768]).to(DEVICE).eval()
teacher_model.load_state_dict(torch.load("checkpoints/depth_anything_v2_vitb.pth", map_location="cpu"))

# === 시간 측정 함수 ===
def measure_time(model, input_tensor, trials=10, upsample=None):
    times = []
    with torch.no_grad():
        for _ in range(trials):
            start = time.perf_counter()
            out = model(input_tensor)
            if upsample:
                out = torch.nn.functional.interpolate(out, size=upsample, mode='bilinear', align_corners=False)
            _ = out.cpu()
            times.append(time.perf_counter() - start)
    return np.mean(times), np.std(times)

# === 예측 수행 ===
with torch.no_grad():
    teacher_pred = teacher_model(input_tensor)
    if teacher_pred.dim() == 3:
        teacher_pred = teacher_pred.unsqueeze(1)
    teacher_pred = torch.nn.functional.interpolate(teacher_pred, size=UPSAMPLE_SIZE, mode='bilinear', align_corners=False)
    teacher_pred = teacher_pred.squeeze().detach().cpu().numpy()

    student_pred = student_model(input_tensor)
    student_pred = torch.nn.functional.interpolate(student_pred, size=UPSAMPLE_SIZE, mode='bilinear', align_corners=False)
    student_pred = student_pred.squeeze().detach().cpu().numpy()

# === 컬러맵 시각화 함수 ===
def save_colored_depth(depth_np, save_path, cmap='plasma'):
    depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
    colormapped = plt.get_cmap(cmap)(depth_norm)[:, :, :3]  # RGBA → RGB
    colormapped = (colormapped * 255).astype(np.uint8)
    colormapped_bgr = cv2.cvtColor(colormapped, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, colormapped_bgr)

error_map = np.abs(teacher_pred - student_pred)

# === 저장 ===
os.makedirs("comparison_output", exist_ok=True)
save_colored_depth(teacher_pred, "comparison_output/depth_anything_v2_vitb_colored.png")
save_colored_depth(student_pred, "comparison_output/Student_colored.png")
save_colored_depth(error_map, "comparison_output/error_map_colored.png", cmap="inferno")