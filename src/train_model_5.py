import os
import json
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from student_model_3 import DepthEstimationMobileNetV2
from model_dataset_4 import JsonDepthDataset

# 디바이스 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transform 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

# ✅ 사용자 정의 학습 데이터 수 및 DDAD:ETRI 비율 설정
TOTAL_SIZE = 5000       # 전체 사용할 학습 샘플 수
ETRI_RATIO = 0.2         # ETRI 비율 (예: 20%)

# 전체 학습 JSON 로딩
with open("dataset_paths/dataset_paths_train.json", "r") as f:
    full_data = json.load(f)

# 경로 기준으로 DDAD/ETRI 구분
etri_data = [item for item in full_data if "/dataset/rgb" in item["rgb"]]
ddad_data = [item for item in full_data if "/output/rgb" in item["rgb"]]

random.shuffle(etri_data)
random.shuffle(ddad_data)

etri_count = int(TOTAL_SIZE * ETRI_RATIO)
ddad_count = TOTAL_SIZE - etri_count

etri_sample = etri_data[:min(etri_count, len(etri_data))]
ddad_sample = ddad_data[:min(ddad_count, len(ddad_data))]

combined_data = etri_sample + ddad_sample
random.shuffle(combined_data)

# 임시 저장
with open("dataset_paths/dataset_paths_filtered.json", "w") as f:
    json.dump(combined_data, f, indent=2)

# 최종 Dataset 로딩
dataset = JsonDepthDataset("dataset_paths/dataset_paths_filtered.json", transform=transform)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 모델 정의
model = DepthEstimationMobileNetV2().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 학습 설정
max_epochs = 10
os.makedirs("trained_models", exist_ok=True)

# 학습 루프
for epoch in range(1, max_epochs + 1):
    model.train()
    total_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", ncols=100)
    for rgb, depth in pbar:
        rgb, depth = rgb.to(device), depth.to(device)

        pred = model(rgb)
        pred_upsampled = F.interpolate(pred, size=depth.shape[2:], mode='bilinear', align_corners=False)

        loss = F.l1_loss(pred_upsampled, depth)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} 평균 손실: {avg_loss:.4f}")

# 모델 저장
save_path = "trained_models/depth_student_mobilenetv2.pth"
torch.save(model.state_dict(), save_path)
print(f"모델 저장 완료: {save_path}")