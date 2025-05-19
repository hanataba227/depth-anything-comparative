import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from student_model_3 import DepthEstimationMobileNetV2
from model_dataset_4 import DepthDataset

from tqdm import tqdm

# ✅ 디바이스 확인 (MPS → CUDA → CPU 순서)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"✅ Using device: {device}")

# ✅ Transform 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=0.5, std=0.5)
])

# ✅ 데이터셋 경로 설정
ddad_path = "vis_depth/ddad"       # DDAD: png + npy 모두 존재
etri_path = "dataset"              # ETRI: dataset/images + dataset/pseudo_depth

# ✅ 데이터셋 로드
train_dataset = DepthDataset(
    ddad_root=ddad_path,
    etri_root=etri_path,
    transform=transform,
    max_ddad=5000,
    max_etri=1200
)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

print(f"📊 Total training samples: {len(train_dataset)}")

# ✅ 모델 정의
model = DepthEstimationMobileNetV2().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ✅ 학습 설정
max_epochs = 10
early_stop_threshold = 0.001
early_stop_patience = 5
prev_loss = float('inf')
patience_counter = 0
os.makedirs("trained_models", exist_ok=True)

# ✅ 학습 루프
for epoch in range(1, max_epochs + 1):
    model.train()
    total_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", ncols=100)
    for i, (rgb, depth) in enumerate(pbar):
        rgb, depth = rgb.to(device), depth.to(device)

        # ✅ 크기 불일치 또는 차원 에러 체크
        if depth.ndim != 3 or rgb.ndim != 4 or depth.shape[1:] != (518, 518):
            #print(f"❌ Invalid depth shape: {depth.shape}")
            continue

        pred = model(rgb)
        pred_upsampled = F.interpolate(pred, size=depth.shape[1:], mode='bilinear', align_corners=False)

        loss = F.l1_loss(pred_upsampled, depth)

        if epoch == 1 and i == 0:  # 첫 배치에서만 출력
            print(f"▶ depth stats: min={depth.min().item():.4f}, max={depth.max().item():.4f}, std={depth.std().item():.4f}")
            print(f"▶ pred stats: mean={pred_upsampled.mean().item():.4f}")
            print(f"▶ loss: {loss.item():.6f}")

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
            print(f"🛑 Early stopping at epoch {epoch}")
            break
    else:
        patience_counter = 0

    prev_loss = avg_loss

# ✅ 모델 저장
save_path = "trained_models/depth_student_mobilenetv2.pth"
torch.save(model.state_dict(), save_path)
print(f"📦 모델 저장 완료: {save_path}")