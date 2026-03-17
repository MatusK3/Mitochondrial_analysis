import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from fetch_data import fetch_ROI_dataset




TARGET_SIZE = (112, 112) 
augmt = T.Compose([
    T.Resize(TARGET_SIZE)
])


def calculate_dataset_stats(dataset, batch_size=64, num_workers=4):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    cnt = 0
    fst_moment = torch.empty(1)
    snd_moment = torch.empty(1)

    print("Calculating Mean and Std...")
    for images, _ in tqdm(loader):
        # images shape: [B, C, H, W]
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        
        # Sum of pixels
        sum_ = torch.sum(images, dim=[0, 2, 3])
        # Sum of squared pixels
        sum_of_square = torch.sum(images**2, dim=[0, 2, 3])
        
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        
        cnt += nb_pixels

    mean = fst_moment
    std = torch.sqrt(snd_moment - fst_moment**2)
    
    return mean.item(), std.item()


train_set_for_stats = fetch_ROI_dataset("train", augmt)

mean, std = calculate_dataset_stats(train_set_for_stats)

print(f"\nResults for your config:")
print(f"Mean: {mean:.6f}")
print(f"Std:  {std:.6f}")