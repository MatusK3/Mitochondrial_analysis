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


def examine_reconstruction(exp: ExperimentLoader, dataloader, plot_results=False):
    # get class names
    DATSET_PATH = "data/mito"
    config_path = f"{DATSET_PATH}/config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    int_to_cls_mapping = {val: key for key, val in config["class_to_int_mapping"].items()}

    name = exp.name
    device = exp.device
    model = exp.model

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x, y = batch[0].to(device), batch[1]

            recon, _, _ = model(x)
            
            x = x.cpu().numpy()
            recon = F.sigmoid(recon).cpu().numpy()
            
            rows = 3
            fig_size = 4
            for j in range(0, len(recon), rows):
                fig, axes = plt.subplots(rows, 2, figsize=(fig_size*2, fig_size*rows))
                if rows == 1: axes = [axes]
                for i in range(rows):
                    input = np.moveaxis(x[i+j], 0, -1)
                    output = np.moveaxis(recon[i+j], 0, -1)

                    axes[i][0].imshow(input)
                    axes[i][0].set_title(f"{int_to_cls_mapping[y[j+i].item()]}")
                    # axes[0].axis("off")

                    axes[i][1].imshow(output)
                    axes[i][1].set_title(f"{int_to_cls_mapping[y[j+i].item()]}, reconstruction")
                    # axes[1].axis("off")

                plt.tight_layout()
                plt.show()

            
    



if __name__ == "__main__":
    # results = []
    # dataloader_test = fetch_test_dataloader()
    from fetch_data import fetch_train_val_dataloaders
    train, dataloader_test = fetch_train_val_dataloaders(batch_size = 3)
    for exp in experiments:
        exp = ExperimentLoader(exp)
        exp.load_model()
        examine_reconstruction(exp, dataloader_test)

    

    # n=5
    # top_n = sorted(results, key=lambda x: x[2], reverse=True)[:n] # BY ACCURACY
    # for key, avg_loss, acc in top_n:
    #     print(key, acc)
    

# resnet9; Weighted F1: 0.6624
# resnet18; Weighted F1: 0.6776
# resnet34; Weighted F1: 0.5785