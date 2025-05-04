import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from student_model_3 import DepthEstimationMobileNetV2
from model_dataset_4 import DepthDataset
from tqdm import tqdm

# 디바이스 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transform 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

# Dataset 로딩
train_dataset = DepthDataset("dataset", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 모델 정의
model = DepthEstimationMobileNetV2().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 학습 설정
max_epochs = 5
early_stop_threshold = 0.001
early_stop_patience = 5
prev_loss = float('inf')
patience_counter = 0
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

    # Early stopping
    if abs(prev_loss - avg_loss) < early_stop_threshold:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break
    else:
        patience_counter = 0

    prev_loss = avg_loss

# 모델 저장
save_path = "trained_models/depth_student_mobilenetv2.pth"
torch.save(model.state_dict(), save_path)
print(f"모델 저장 완료: {save_path}")
