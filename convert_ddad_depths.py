import os
import json
import numpy as np
from tqdm import tqdm

# JSON 경로
json_path = "/Users/gang-yeonghui/Documents/4. DSC 졸업과제/Depth-Anything-V2-main/archive/DDAD/ddad_train_val/ddad.json"

# JSON 파일 로드
with open(json_path, 'r') as f:
    ddad_data = json.load(f)

# "0"과 "1" 둘 다 처리
for split_key in ddad_data["scene_splits"]:
    scene_list = ddad_data["scene_splits"][split_key]["filenames"]

    for scene_path in scene_list:
        fileNum = scene_path.split('/')[0]

        # 디렉토리 경로 구성
        lidar_dir = f"/Users/gang-yeonghui/Documents/4. DSC 졸업과제/Depth-Anything-V2-main/archive/DDAD/ddad_train_val/{fileNum}/point_cloud/LIDAR"
        save_dir = f"/Users/gang-yeonghui/Documents/4. DSC 졸업과제/Depth-Anything-V2-main/archive/ddad_depth_maps/{fileNum}"
        os.makedirs(save_dir, exist_ok=True)

        if not os.path.exists(lidar_dir):
            print(f"[Skip] {lidar_dir} 없음")
            continue

        # 변환 수행
        for filename in tqdm(os.listdir(lidar_dir), desc=f"{fileNum} 변환 중"):
            if filename.endswith(".npz"):
                npz_path = os.path.join(lidar_dir, filename)
                npy_path = os.path.join(save_dir, filename.replace(".npz", ".npy"))

                try:
                    data = np.load(npz_path)
                    if 'data' in data:
                        np.save(npy_path, data['data'])
                    elif len(data.files) == 1:
                        np.save(npy_path, data[data.files[0]])
                    else:
                        np.save(npy_path, dict(data))
                except Exception as e:
                    print(f"[Error] {fileNum}/{filename} 변환 실패: {e}")