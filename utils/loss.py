# utils/loss.py
# metric depth 기반 학습을 위한 손실 함수
# L1 + scale-invariant log RMSE

import torch

def scale_invariant_loss(pred, target):
    diff = torch.log(pred + 1e-6) - torch.log(target + 1e-6)
    return torch.sqrt((diff ** 2).mean() - 0.85 * (diff.mean()) ** 2)

def combined_loss(pred, target):
    l1 = torch.abs(pred - target).mean()
    silog = scale_invariant_loss(pred, target)
    return l1 + 0.5 * silog