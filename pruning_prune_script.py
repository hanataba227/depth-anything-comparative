import argparse
import torch
import torch.nn.utils.prune as prune
from depth_anything_v2.dpt import DepthAnythingV2
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conv_unstruct", type=float, default=0.2)
    parser.add_argument("--output_path", type=str, default="checkpoints/pruned_depth_anything_v2_vitb.pth")
    parser.add_argument("--kept_indices_path", type=str, default="kept_indices/encoder_conv1.pt")
    return parser.parse_args()

def prune_patch_embed_only(model, amount=0.2):
    """encoder의 patch_embed.proj에만 pruning 수행"""
    proj = model.pretrained.patch_embed.proj
    print(f"Before pruning: {proj.weight.shape}")
    prune.l1_unstructured(proj, name='weight', amount=amount)
    kept_indices = (proj.weight_mask.sum(dim=(1, 2, 3)) > 0).nonzero(as_tuple=False).squeeze(1)
    print(f"After pruning: {kept_indices.numel()} filters kept out of {proj.weight.shape[0]}")
    return proj, kept_indices

if __name__ == '__main__':
    args = parse_args()
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("📦 모델 로드 중...")
    model = DepthAnythingV2(encoder='vitb', features=128, out_channels=[96,192,384,768]).to(DEVICE)
    ckpt = torch.load('weights/depth_anything_v2_vitb.pth', map_location=DEVICE)
    model.load_state_dict(ckpt)

    print("🔨 patch_embed.proj pruning 수행 중...")
    proj, kept_indices = prune_patch_embed_only(model, amount=args.conv_unstruct)

    os.makedirs(os.path.dirname(args.kept_indices_path), exist_ok=True)
    torch.save(kept_indices, args.kept_indices_path)
    print(f"✅ Kept indices saved to {args.kept_indices_path} ({len(kept_indices)} filters)")

    prune.remove(proj, 'weight')
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(model.state_dict(), args.output_path)
    print(f"✅ Pruned model saved to {args.output_path}")