
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
    # *[
    #     ExperimentConfig(
    #         name = model_name,
    #         result_sub_dir = model_name,
    #         model_fetcher=partial(model_fetcher, num_classes=num_classes),
    #         data_fetcher=partial(fetch_data.fetch_train_val_dataloaders, batch_size=16),
    #         loss_function=torch.nn.CrossEntropyLoss,
    #         optimizer_factory=partial(torch.optim.Adam, lr=1e-3),
    #         scheduler_factory=partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=e), #None,
    #         epochs=e,
    #         prune_iter = 15
    #     )
    #     for model_fetcher, model_name in [[models.get_resnet9, "resnet9"], [models.get_resnet18, "resnet18"], [models.get_resnet34, "resnet34"]]
    #     for e in [100]
    # ]
    *[
        ExperimentConfig(
            name = model_name,
            result_sub_dir = model_name,
            model_fetcher=partial(model_fetcher, num_classes=num_classes),
            data_fetcher=partial(fetch_data.fetch_train_val_dataloaders, batch_size=16),
            loss_function=torch.nn.CrossEntropyLoss,
            optimizer_factory=partial(torch.optim.Adam, lr=1e-3),
            scheduler_factory=partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=e), #None,
            epochs=e,
            prune_iter = 1
        )
        for model_fetcher, model_name in [[models.get_resnet34, "resnet34"]]
        for e in [100]
    ]
]