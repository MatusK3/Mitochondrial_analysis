import torch
import pandas as pd
import os

from experiments.experiments_config import RESULTS_SAVE_DIR, ExperimentConfig

from pathlib import Path
import json

def get_dataset_class_weights():
    DATASET_PATH = Path("data/mito")
    config_path = Path.joinpath(DATASET_PATH, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    class_distribution = config["split_class_distribution"]["train"]
    class_to_int_mapping = config["class_to_int_mapping"]

    class_counts  = torch.zeros(len(class_to_int_mapping))
    for class_name, idx in class_to_int_mapping.items():
        class_counts[idx] = class_distribution.get(class_name, 0)

    weights = 1.0 / class_counts.float()
    weights = weights / weights.sum()

    return weights

class ExperimentLoader:
    def __init__(self, config, device=None):
        self.EXP: ExperimentConfig  = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)

        self._train_loader, self._val_loader = None, None

        self.initialize()

    def initialize(self):
        self.name = self.EXP.name
        self.result_sub_dir = self.EXP.result_sub_dir

        results_dir = f"{RESULTS_SAVE_DIR}/{self.result_sub_dir}"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        self.prune_iter = self.EXP.prune_iter
        self._model_paths, self._training_log_paths = [], []
        for i in range(self.prune_iter):
            self._model_paths.append(f"{results_dir}/p{i}_{self.name}.pth")
            self._training_log_paths.append(f"{results_dir}/p{i}_{self.name}_training_log.csv")
        self.evaluations_path = f"{results_dir}/{self.name}_test_eval.csv"

        self.model = self.EXP.model_fetcher()
        self.model.to(self.device)

        weight = get_dataset_class_weights()
        self.loss_fn = self.EXP.loss_function(weight = weight.to(self.device))
        # self._optimizer = None #self.EXP.optimizer_factory(self.model.parameters())
        # self.scheduler = None

        # if self.EXP.scheduler_factory is not None:
        #     self.scheduler = self.EXP.scheduler_factory(self._optimizer)
        # else:

        self.epochs = self.EXP.epochs

    def load_model(self, prune_iteration=0):
        state = torch.load(self._model_paths[prune_iteration], map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)


    def get_optimizer_and_scheduler(self): # reinitialize optimizer, so if used during pruning, its reseted each time
        optimizer = self.EXP.optimizer_factory(self.model.parameters())
        if self.EXP.scheduler_factory is not None:
            return optimizer, self.EXP.scheduler_factory(optimizer)
        else:
            return optimizer, None

    def get_model_path(self, prune_iteration=0):
        return self._model_paths[prune_iteration]
    
    def get_training_log_path(self, prune_iteration=0):
        return self._training_log_paths[prune_iteration]
    
    def initialize_train_val_dataset_loader(self):
        self._train_loader, self._val_loader = self.EXP.data_fetcher()

    def get_train_loader(self):
        if self._train_loader is None: self.initialize_train_val_dataset_loader()
        return self._train_loader

    def get_val_loader(self):
        if self._val_loader is None: self.initialize_train_val_dataset_loader()
        return self._val_loader


if __name__ == "__main__":
    print("weights:", get_dataset_class_weights())