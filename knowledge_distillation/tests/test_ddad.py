import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from ddad import DDADDepthDataset

dataset = DDADDepthDataset(root_dir="C:/project/depth-anything/DDAD", cameras=['CAMERA_01'], img_size=(384, 384))
print(f"Dataset size: {len(dataset)}")
print(f"Dataset root_dir: {dataset.root_dir}")

dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

images, depths = next(iter(dataloader))

def unnormalize(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img * std + mean
    return img

images, depths = next(iter(dataloader))

unnorm_img = unnormalize(images[0]).permute(1, 2, 0).numpy()
depth_map = depths[0].numpy()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('RGB Image')
plt.imshow(unnorm_img)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Generated Depth')
plt.imshow(depth_map, cmap='plasma')
plt.axis('off')

plt.show()
