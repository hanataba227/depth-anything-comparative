import torch
from depth_anything_v2.dpt import DepthAnythingV2_Rebuilt
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pruned_ckpt_path", type=str, default="checkpoints/pruned_depth_anything_v2_vitb.pth")
    parser.add_argument("--kept_indices_path", type=str, default="kept_indices/encoder_conv1.pt")
    parser.add_argument("--rebuilt_ckpt_path", type=str, default="checkpoints/rebuilt_depth_anything_v2_vitb.pth")
    return parser.parse_args()

def main():
    args = parse_args()
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    kept_indices = torch.load(args.kept_indices_path)
    print(f"🔨 Building rebuilt model with {len(kept_indices)} patch_embed filters...")
    rebuilt_model = DepthAnythingV2_Rebuilt(kept_indices_path=args.kept_indices_path).to(DEVICE)

    print("📦 Loading weights from pruned model with filtering...")
    full_state = torch.load(args.pruned_ckpt_path, map_location=DEVICE)
    rebuilt_state = rebuilt_model.state_dict()

    filtered_state = {
        k: v for k, v in full_state.items()
        if k in rebuilt_state and v.shape == rebuilt_state[k].shape
    }

    missing, unexpected = rebuilt_model.load_state_dict(filtered_state, strict=False)
    print(f"✅ Filtered load complete. Loaded: {len(filtered_state)} / {len(rebuilt_state)}")
    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    os.makedirs(os.path.dirname(args.rebuilt_ckpt_path), exist_ok=True)
    torch.save(rebuilt_model.state_dict(), args.rebuilt_ckpt_path)
    print(f"✅ Saved rebuilt model to: {args.rebuilt_ckpt_path}")

if __name__ == "__main__":
    main()