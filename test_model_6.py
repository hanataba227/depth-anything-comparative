import os
import time
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
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

# === 추론 시간 비교 ===
teacher_time, teacher_std = measure_time(teacher_model, input_tensor, NUM_TRIALS)
student_time, student_std = measure_time(student_model, input_tensor, NUM_TRIALS, upsample=UPSAMPLE_SIZE)
improvement = ((teacher_time - student_time) / teacher_time) * 100

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

# === 정규화 함수 ===
def to_uint8(depth):
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return (depth * 255).astype(np.uint8)

# === 시각화용 변환 ===
teacher_map = to_uint8(teacher_pred)
student_map = to_uint8(student_pred)
error_map = np.abs(teacher_map.astype(np.float32) - student_map.astype(np.float32)).astype(np.uint8)

# === 오차 지표 계산 ===
mae = np.mean(np.abs(teacher_pred - student_pred))
rmse = np.sqrt(np.mean((teacher_pred - student_pred) ** 2))

# === 저장 ===
os.makedirs("comparison_output", exist_ok=True)
cv2.imwrite("comparison_output/original.jpg", cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
cv2.imwrite("comparison_output/depth_anything_v2_vitb.png", teacher_map)
cv2.imwrite("comparison_output/ETRI.png", student_map)
cv2.imwrite("comparison_output/error_map.png", error_map)

# === 출력 ===
print(f"\n결과 요약")
print(f"depth_anything_v2_vitb 평균 추론 시간: {teacher_time:.4f}s ± {teacher_std:.4f}")
print(f"에트리 모델 평균 추론 시간: {student_time:.4f}s ± {student_std:.4f}")
print(f"응답속도 개선률: {improvement:.2f}%")
print(f"MAE (Mean Absolute Error): {mae:.4f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
