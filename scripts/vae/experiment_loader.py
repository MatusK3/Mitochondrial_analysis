import torch
import os

from experiments_config import RESULTS_SAVE_DIR, ExperimentConfig



class ExperimentLoader:
    def __init__(self, config, device=None):
        self.EXP: ExperimentConfig  = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self._train_loader, self._val_loader = None, None

        self.initialize()

    def initialize(self):
        self.name = self.EXP.name
        self.result_sub_dir = self.EXP.result_sub_dir

        results_dir = f"{RESULTS_SAVE_DIR}/{self.result_sub_dir}"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        self._model_path= f"{results_dir}/{self.name}.pth"
        self._training_log_path = f"{results_dir}/{self.name}_training_log.csv"

        self.model = self.EXP.model_fetcher()
        self.model.to(self.device)

        self.loss_fn = self.EXP.loss_function

        self.epochs = self.EXP.epochs

    def load_model(self):
        state = torch.load(self._model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)

    def get_optimizer_and_scheduler(self): # reinitialize optimizer, so if used during pruning, its reseted each time
        optimizer = self.EXP.optimizer_factory(self.model.parameters())
        if self.EXP.scheduler_factory is not None:
            return optimizer, self.EXP.scheduler_factory(optimizer)
        else:
            return optimizer, None

    def get_model_path(self):
        return self._model_path
    
    def get_training_log_path(self):
        return self._training_log_path
    
    def initialize_train_val_dataset_loader(self):
        self._train_loader, self._val_loader = self.EXP.data_fetcher()

    def get_train_loader(self):
        if self._train_loader is None: self.initialize_train_val_dataset_loader()
        return self._train_loader

    def get_val_loader(self):
        if self._val_loader is None: self.initialize_train_val_dataset_loader()
        return self._val_loader



if __name__ == "__main__":
    ...





