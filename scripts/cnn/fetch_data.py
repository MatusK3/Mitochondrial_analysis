import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import h5py
import os
import numpy as np

from typing import Tuple


class FlattenedROIDataset(Dataset):
    def __init__(self, csv_path: str, config_path: str, data_dir: str, split_name: str="train", transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Load labels and config
        df = pd.read_csv(csv_path)
        with open(config_path, 'r') as f:
            config = json.load(f)

        class_mapping = config["class_mapping"]
        class_to_int = config["class_to_int_mapping"]
        
        allowed_scenes = config["dataset_split"][split_name]
        filtered_df = df[df['scene'].isin(allowed_scenes)]
        scene_df = filtered_df[["scene", "environment"]].drop_duplicates()

        self.roi_samples = []
        for _, row in scene_df.iterrows():
            scene_id = row["scene"]

            env_name = row["environment"] 
            group_name = class_mapping[env_name]
            label_idx = class_to_int[group_name]

            roi_path = f"{self.data_dir}/{scene_id}.h5"

            # count ROIs in h5 file
            with h5py.File(roi_path, "r") as f:
                for key in sorted(f.keys()):
                    self.roi_samples.append({
                        "path": roi_path,
                        "key": key,
                        "label": label_idx
                    })
            
        self.classes = list(class_to_int.keys())
    
    def __len__(self):
        return len(self.roi_samples)

    def __getitem__(self, idx):
        sample_info = self.roi_samples[idx]
        
        with h5py.File(sample_info['path'], 'r') as f:
            roi = f[sample_info['key']][()]

        roi = roi.astype(np.float32) / 65535.0 # Scale uint16 (0-65535) to float32 (0.0-1.0) manually
        roi = roi[np.newaxis, ...] # add new axis. from grayscale (H, W) into (1, H, W)
        img_tensor = torch.from_numpy(roi).float()
        if self.transform:
            img_tensor = self.transform(img_tensor)

        label_tensor = torch.tensor(sample_info['label'], dtype=torch.long)

        return img_tensor, label_tensor
    

def fetch_ROI_dataset(split_type:str = "train", transformation=None):
    DATSET_PATH = "data/mito"
    csv_path = f"{DATSET_PATH}/mito.csv"
    config_path = f"{DATSET_PATH}/config.json"
    DATA_DIR = "results/roi"
    return FlattenedROIDataset(csv_path, config_path, DATA_DIR, split_type, transform=transformation)



# MEAN = 0.020517
# STD = 0.018828
MEAN =0.027762
STD = 0.025419
norm_background = (0 - MEAN)/STD # background org zero => normalized background
TARGET_SIZE = (112, 112)
transform_train = T.Compose([
    T.RandomResizedCrop(TARGET_SIZE, scale=(0.6, 1.0)), 
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=180),
    T.RandomAffine(degrees=0, shear=10),

    T.RandomApply([
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
    ], p=0.7),

    T.RandomApply([
        T.GaussianBlur(kernel_size=3)
    ], p=0.3),

    T.Normalize(mean=MEAN, std=STD),

    T.RandomErasing(p=0.5, scale=(0.02, 0.2), value=norm_background), 
])

transform_test = T.Compose([
    T.Resize(TARGET_SIZE),
    T.Normalize(mean=MEAN, std=STD),
])



def fetch_train_val_dataloaders(batch_size: int=64, transform=transform_train) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    trainset = fetch_ROI_dataset("train", transformation=transform)
    valset = fetch_ROI_dataset("val", transformation=transform_test)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def fetch_test_dataloader(batch_size: int = 100):
    testset = fetch_ROI_dataset("test", transformation=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return test_loader


def fetch_mini(num_of_samples: int = 32, random_seed=42) -> torch.utils.data.DataLoader:
    dataset = fetch_ROI_dataset("train", transformation=transform_train)

    rng = np.random.default_rng(seed=random_seed)
    random_selected_samples = rng.choice(len(dataset), size=num_of_samples, replace=False)

    mini = torch.utils.data.Subset(dataset, random_selected_samples)
    mini_loader = torch.utils.data.DataLoader(mini,batch_size=num_of_samples,shuffle=True)
    return mini_loader








if __name__ == "__main__":
    import matplotlib.pyplot as plt 

    # trainset = fetch_ROI_dataset("train", transformation=transform_train)
    # # valset = fetch_ROI_dataset("val")

    # classes = trainset.classes

    # previewloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
    DATSET_PATH = "data/mito"
    config_path = f"{DATSET_PATH}/config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    classes = list(config["class_to_int_mapping"].keys())

    previewloader = fetch_test_dataloader(batch_size = 1)
    # train, previewloader = fetch_train_val_dataloaders(batch_size = 1)
    # previewloader, val = fetch_train_val_dataloaders(batch_size = 1, transform=transform_train)

    for i, sample in enumerate(previewloader):
        # if i > 10: break
        # if i == 0: continue
        

        x, y = sample
        # if y.item() != 2: continue

        print(x.shape, x.min(), x.max())

        img = np.moveaxis(x.numpy()[0], 0, -1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        # plt.title(f"{y}: {classes[y.item()]}")
        plt.title(f"{classes[y.item()]}")
        plt.show()

        # print(classes[y.item()])