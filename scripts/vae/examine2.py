import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import torch.nn.functional as F

from experiment_loader import ExperimentLoader
from fetch_data import fetch_test_dataloader
from experiments_config import experiments

import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def examine_reconstruction(exps, dataloader, plot_results=False):
    # get class names
    DATSET_PATH = "data/mito"
    config_path = f"{DATSET_PATH}/config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    int_to_cls_mapping = {val: key for key, val in config["class_to_int_mapping"].items()}
    device = 'cuda'
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch[0].to(device), batch[1]
            

            batch_recons = []
            exps_names = []
            for exp in exps:
                exp = ExperimentLoader(exp)
                exp.load_model()

                model = exp.model
                model.eval()
                recon, _, _ = model(x)
                
                
                recon = F.sigmoid(recon).cpu().numpy()
                batch_recons.append(recon)
                exps_names.append(exp.name)
            x = x.cpu().numpy()

            for i in range(len(y)):
                plt.imshow(np.moveaxis(x[i], 0, -1))
                plt.title(int_to_cls_mapping[y[i].item()])
                rows, cols = 3, 6
                fig_size = 4
                fig, axes = plt.subplots(rows, cols, figsize=(fig_size*cols, fig_size*rows))
                axes = axes.flatten()
                for j in range(len(exps)):
                    axes[j].imshow(np.moveaxis(batch_recons[j][i], 0, -1))
                    axes[j].set_title(exps_names[j])

                plt.tight_layout()
                plt.show()


if __name__ == "__main__":
    # results = []
    # dataloader_test = fetch_test_dataloader()
    from fetch_data import fetch_train_val_dataloaders
    train, dataloader_test = fetch_train_val_dataloaders(batch_size = 3)
    examine_reconstruction(experiments, dataloader_test)

    

    # n=5
    # top_n = sorted(results, key=lambda x: x[2], reverse=True)[:n] # BY ACCURACY
    # for key, avg_loss, acc in top_n:
    #     print(key, acc)
    

# resnet9; Weighted F1: 0.6624
# resnet18; Weighted F1: 0.6776
# resnet34; Weighted F1: 0.5785