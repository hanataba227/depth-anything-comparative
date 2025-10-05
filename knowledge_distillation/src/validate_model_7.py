import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from model_dataset_4 import JsonDepthDataset
from student_model_3 import DepthEstimationMobileNetV2

sys.path.append("C:/project/Depth-Anything-V2")
from depth_anything_v2.dpt import DepthAnythingV2

# 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"검증용 디바이스: {DEVICE}")

# Transform 정의 (학습과 동일)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

# 검증 데이터 최대 개수 설정
VAL_SAMPLE_SIZE = 300

# 검증용 데이터 로딩
val_dataset = JsonDepthDataset("dataset_paths/dataset_paths_val.json", transform=transform)
val_subset = Subset(val_dataset, list(range(min(VAL_SAMPLE_SIZE, len(val_dataset)))))
val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)

# 모델 로딩
student_model = DepthEstimationMobileNetV2().to(DEVICE).eval()
student_model.load_state_dict(torch.load("trained_models/depth_student_mobilenetv2_3.pth"))

teacher_model = DepthAnythingV2(encoder="vitb", features=128, out_channels=[96, 192, 384, 768]).to(DEVICE).eval()
teacher_model.load_state_dict(torch.load("checkpoints/depth_anything_v2_vitb.pth", map_location="cpu"))

# Scale & Shift alignment 함수
def scale_and_shift_align(pred, target):
    mask = (target > 0)
    pred_valid = pred[mask]
    target_valid = target[mask]
    A = np.vstack([pred_valid, np.ones_like(pred_valid)]).T
    result = np.linalg.lstsq(A, target_valid, rcond=None)
    scale, shift = result[0]
    return scale * pred + shift

# Teacher 평가 함수 (정렬 포함)
def evaluate_teacher(model):
    mae_list, rmse_list = [], []
    with torch.no_grad():
        for rgb, depth in tqdm(val_loader, desc="[Teacher] 검증 중"):
            rgb, depth = rgb.to(DEVICE), depth.to(DEVICE)
            pred = model(rgb)
            if pred.dim() == 3:
                pred = pred.unsqueeze(1)
            pred_up = F.interpolate(pred, size=depth.shape[2:], mode='bilinear', align_corners=False)
            pred_np = pred_up.squeeze().cpu().numpy()
            depth_np = depth.squeeze().cpu().numpy()
            aligned_pred = scale_and_shift_align(pred_np, depth_np)
            mae = np.mean(np.abs(aligned_pred - depth_np))
            rmse = np.sqrt(np.mean((aligned_pred - depth_np) ** 2))
            mae_list.append(mae)
            rmse_list.append(rmse)
    return np.mean(mae_list), np.mean(rmse_list)

# Student 평가 함수 (정규화 기반)
def evaluate_student(model):
    mae_list, rmse_list = [], []
    with torch.no_grad():
        for rgb, depth in tqdm(val_loader, desc="[Student] 검증 중"):
            rgb, depth = rgb.to(DEVICE), depth.to(DEVICE)
            pred = model(rgb)
            if pred.dim() == 3:
                pred = pred.unsqueeze(1)
            pred_up = F.interpolate(pred, size=depth.shape[2:], mode='bilinear', align_corners=False)
            pred_np = pred_up.squeeze().cpu().numpy()
            depth_np = depth.squeeze().cpu().numpy()
            # 정규화 기준 비교
            pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)
            depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
            mae = np.mean(np.abs(pred_np - depth_np))
            rmse = np.sqrt(np.mean((pred_np - depth_np) ** 2))
            mae_list.append(mae)
            rmse_list.append(rmse)
    return np.mean(mae_list), np.mean(rmse_list)

# 모델별 평가
student_mae, student_rmse = evaluate_student(student_model)
teacher_mae, teacher_rmse = evaluate_teacher(teacher_model)

# 결과 출력
print("\n정확도 비교 (Depth Estimation)")
print("모델                 MAE       RMSE")
print("-------------------  --------  --------")
print(f"DepthAnything V2     {teacher_mae:.4f}    {teacher_rmse:.4f} (실제 스케일 기준)")
print(f"Student (MobileNet)  {student_mae:.4f}    {student_rmse:.4f} (정규화 기준)")
