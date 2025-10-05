import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from depth_anything_v2.dpt import DepthAnythingV2
from PIL import Image
from glob import glob

# 사용자 정의 Dataset
class DepthDataset(Dataset):
    def __init__(self, img_paths, depth_paths, transform_rgb, transform_depth):
        self.img_paths = img_paths
        self.depth_paths = depth_paths
        self.transform_rgb = transform_rgb
        self.transform_depth = transform_depth

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        depth = Image.open(self.depth_paths[idx])
        return self.transform_rgb(img), self.transform_depth(depth)

# 인자 파싱
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="pruned_model.pth")
    parser.add_argument("--output", type=str, default="finetuned_model.pth")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    return parser.parse_args()

# 학습 함수
def train():
    args = parse_args()
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 모델 초기화 및 프루닝된 weight 로드
    model = DepthAnythingV2(encoder='vitb', features=128, out_channels=[96,192,384,768]).to(DEVICE)
    model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))
    model.train()

    # 실제 경로에 맞게 파일 경로 설정
    img_paths = sorted(glob("C:/project/depth-anything-comparative_2/output/rgb/CAMERA_01/*.png"))
    depth_paths = sorted(glob("C:/project/depth-anything-comparative_2/output/depth_gt_sparse/CAMERA_01/*.png"))

    # transform 정의
    transform_rgb = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    transform_depth = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor()
    ])

    # 데이터셋 및 로더
    dataset = DepthDataset(img_paths, depth_paths, transform_rgb, transform_depth)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = 0
        for imgs, depths in loader:
            imgs, depths = imgs.to(DEVICE), depths.to(DEVICE)
            preds = model(imgs)
            loss = F.l1_loss(preds, depths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), args.output)
    print(f"Fine-tuned model saved to {args.output}")

if __name__ == '__main__':
    train()
