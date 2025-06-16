import math
from functools import lru_cache
import torch

from rhino.nn.utils import windowing

@lru_cache()
def compute_mask_3d(input_size, window_size, shift_size):
    t, h, w = input_size
    pT = int(math.ceil(t / window_size[0])) * window_size[0]
    pH = int(math.ceil(h / window_size[1])) * window_size[1]
    pW = int(math.ceil(w / window_size[2])) * window_size[2]
    
    img_mask = torch.zeros((1, pT, pH, pW, 1))  # 1 D H W 1
    
    cnt = 0
    for d in [slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None)]:
        for h in [slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None)]:
            for w in [slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None)]:
                img_mask[:, d, h, w, :] = cnt
                cnt += 1

    mask_windows = windowing.window_partition_3d(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size[0] * window_size[1] * window_size[2])
    
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    
    return attn_mask

@lru_cache()
def compute_mask_2d(input_size, window_size, shift_size):
    h, w = input_size
    pH = int(math.ceil(h / window_size[0])) * window_size[0]
    pW = int(math.ceil(w / window_size[1])) * window_size[1]
    
    img_mask = torch.zeros((1, pH, pW, 1))  # 1 H W 1
    
    cnt = 0
    for h in [slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None)]:
        for w in [slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None)]:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = windowing.window_partition_2d(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size[0] * window_size[1])
    
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    
    return attn_mask