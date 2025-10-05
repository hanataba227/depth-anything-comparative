import os
import json
import random

def collect_pairs(rgb_root, depth_root):
    pairs = []
    for scene in os.listdir(rgb_root):
        rgb_scene_path = os.path.join(rgb_root, scene)
        depth_scene_path = os.path.join(depth_root, scene)
        if not os.path.isdir(rgb_scene_path):
            continue
        for fname in os.listdir(rgb_scene_path):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            rgb_path = os.path.join(rgb_scene_path, fname)
            depth_path = os.path.join(depth_scene_path, os.path.splitext(fname)[0] + ".png")
            if os.path.exists(depth_path):
                pairs.append({
                    "rgb": rgb_path.replace("\\", "/"),
                    "depth": depth_path.replace("\\", "/")
                })
    return pairs

# ===== 경로 설정 =====
# ETRI
etri_rgb = "C:/project/depth-anything-comparative/dataset/rgb"
etri_depth = "C:/project/depth-anything-comparative/dataset/pseudo_depth"

# DDAD
ddad_rgb = "C:/project/depth-anything-comparative_2/output/rgb"
ddad_depth = "C:/project/depth-anything-comparative_2/output/depth_gt_sparse"

# ===== 데이터 수집 =====
etri_pairs = collect_pairs(etri_rgb, etri_depth)
ddad_pairs = collect_pairs(ddad_rgb, ddad_depth)

# ===== DDAD 검증 비율 분할 =====
val_ratio = 0.3
random.shuffle(ddad_pairs)
split_idx = int(len(ddad_pairs) * (1 - val_ratio))
ddad_train = ddad_pairs[:split_idx]
ddad_val = ddad_pairs[split_idx:]

# ===== 학습 / 검증 세트 구성 =====
train_data = etri_pairs + ddad_train
val_data = ddad_val

# ===== 저장 =====
with open("dataset_paths/dataset_paths_train.json", "w") as f:
    json.dump(train_data, f, indent=2)

with open("dataset_paths/dataset_paths_val.json", "w") as f:
    json.dump(val_data, f, indent=2)

print(f"학습용: {len(train_data)}개")
print(f"검증용: {len(val_data)}개")
