import argparse
import torch
import time
import os
from ptflops import get_model_complexity_info
from depth_anything_v2.dpt import DepthAnythingV2, DepthAnythingV2_Rebuilt

def inspect_model_state_dict(ckpt_path, model):
    state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))
    mismatches = []
    model_keys = set(model.state_dict().keys())
    for key in state_dict.keys():
        if key in model_keys:
            expected_shape = model.state_dict()[key].shape
            loaded_shape = state_dict[key].shape
            if expected_shape != loaded_shape:
                mismatches.append((key, expected_shape, loaded_shape))
        else:
            mismatches.append((key, None, state_dict[key].shape))
    return mismatches

def load_model(ckpt_path, kept_indices_path=None):
    if "rebuilt" in ckpt_path or "finetuned" in ckpt_path:
        if kept_indices_path is None:
            kept_indices_path = "kept_indices/encoder_conv1.pt"
        kept_indices = torch.load(kept_indices_path)
        patch_out = len(kept_indices)
        features = 128
        out_channels = [patch_out, 128, 192, 384]  # head 구조와 일치
        model = DepthAnythingV2_Rebuilt(
            kept_indices_path=kept_indices_path,
            encoder="vitb",
            features=features,
            out_channels=out_channels,
            use_clstoken=True
        )
    else:
        model = DepthAnythingV2(
            encoder="vitb",
            features=128,
            out_channels=[96, 192, 384, 768]
        )
    state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"❌ Failed strict load: {e}\nTrying with strict=False...")
        mismatches = inspect_model_state_dict(ckpt_path, model)
        for key, expected, actual in mismatches:
            print(f"🔎 Mismatch: {key}\n\tExpected: {expected}\n\tGot: {actual}")
        model.load_state_dict(state_dict, strict=False)
    return model

def evaluate_model(ckpt_path, name, input_size=(3, 224, 224), kept_indices_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ckpt_path, kept_indices_path=kept_indices_path).to(device)
    model.eval()

    dummy_input = torch.randn(1, *input_size).to(device)

    try:
        macs, params = get_model_complexity_info(
            model, input_size, as_strings=False, print_per_layer_stat=False, verbose=False
        )
        macs /= 1e9
        params /= 1e6
    except Exception as e:
        print(f"⚠️ ptflops failed for {name}: {e}")
        macs = -1
        params = sum(p.numel() for p in model.parameters()) / 1e6

    try:
        start_time = time.time()
        with torch.no_grad():
            _ = model(dummy_input)
        latency = time.time() - start_time
    except Exception as e:
        print(f"⚠️ Forward pass failed for {name}: {e}")
        latency = -1

    mem_allocated = torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else -1

    return {
        "model": name,
        "params": round(params, 2),
        "file_size": os.path.getsize(ckpt_path),
        "flops": f"{macs:.2f} GMac" if macs != -1 else "N/A",
        "latency_s": round(latency, 4) if latency != -1 else "N/A",
        "peak_memory": mem_allocated,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True,
                        help="List of ckpt_path:name pairs to compare")
    parser.add_argument("--kept_indices_path", type=str, default="kept_indices/encoder_conv1.pt",
                        help="Path to kept_indices for rebuilt/finetuned models")
    parser.add_argument("--input_size", type=int, nargs=3, default=[3, 224, 224],
                        help="Input size as 3 integers: C H W")
    args = parser.parse_args()

    input_size = tuple(args.input_size)
    results = [
        evaluate_model(p, n, input_size, kept_indices_path=args.kept_indices_path)
        for p, n in [m.split(":") for m in args.models]
    ]

    for r in results:
        print(
            f"\n📊 {r['model']}\n"
            f"Params: {r['params']}M\n"
            f"Size: {r['file_size']} bytes\n"
            f"FLOPs: {r['flops']}\n"
            f"Latency: {r['latency_s']}s\n"
            f"Peak Memory: {r['peak_memory']} bytes\n"
        )

if __name__ == "__main__":
    main()