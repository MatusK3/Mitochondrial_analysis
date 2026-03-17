
from functools import partial
from dataclasses import dataclass
import torch
import os
import json

import models
import fetch_data
RESULTS_SAVE_DIR = "results/cnn/"

dataset_path = "data/mito/"
data_structure_path = f"{dataset_path}/mito.csv"
dataset_config_path = f"{dataset_path}/config.json"
with open(dataset_config_path, 'r') as f:
    config = json.load(f)
num_classes = len(config["class_to_int_mapping"])

@dataclass
class ExperimentConfig:
    name: str
    result_sub_dir: str
    model_fetcher: callable
    data_fetcher: callable  
    loss_function: callable
    optimizer_factory: callable
    scheduler_factory: callable = None
    epochs: int = 20,
    prune_iter: int = 15,


experiments = [
    *[
        ExperimentConfig(
            name = f"ld{ld}_e{e}",
            result_sub_dir = model_name,
            model_fetcher=partial(model_fetcher, latent_dim=ld, blocks_per_layer=2),
            # data_fetcher=partial(fetch_data.fetch_train_val_dataloaders, batch_size=32),
            data_fetcher=partial(fetch_data.fetch_big_dataloader, batch_size=32),
            loss_function=models.vae_loss,
            optimizer_factory=partial(torch.optim.Adam, lr=5e-5),
            scheduler_factory=partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=e), #None,
            epochs=e
        )
        for model_fetcher, model_name in [[models.VAE, "VAE"]]
        for ld in [256]
        for e in [1000, 500]
    ]
]