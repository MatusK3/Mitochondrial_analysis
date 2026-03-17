import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt 

from fetch_data import fetch_test_dataloader
from experiments.experiments_config import experiments
from experiment_loader import ExperimentLoader


def find_target_layer(model):
    # Layers we usually want to target for Grad-CAM
    valid_types = (
        nn.Conv2d, 
        # nn.BatchNorm2d, 
        # nn.ReLU
    ) 
    for name, module in reversed(list(model.named_modules())):        
        if isinstance(module, valid_types):
            return [module]
    return None


if __name__ == "__main__":
    # load class names
    DATSET_PATH = "data/mito"
    config_path = f"{DATSET_PATH}/config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    classes = list(config["class_to_int_mapping"].keys())

    test_dataloader: torch.utils.data.DataLoader = fetch_test_dataloader(batch_size = 1)

    for experiment in experiments:#[1:2]:
        exp = ExperimentLoader(experiment)
        
        for i, batch in enumerate(test_dataloader):
            x, y = batch[0].to(exp.device), batch[1].to(exp.device) # x is graylevel image with values normalized around 0
            y_label = y.item()
            

            if y != 2: continue
            
            fig, axes = plt.subplots(3, 5, figsize=(20, 12))
            axes = axes.flatten()

            img_np = x.squeeze().cpu().numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min()) # Min-max scaling
            rgb_img = np.stack([img_np]*3, axis=-1)

            for prune_iteration in range(exp.prune_iter):
                exp.load_model(prune_iteration)
                model = exp.model
                model.eval()

                out = model(x)
                predicted = torch.argmax(out, dim=-1)

                target_layers = find_target_layer(model)
                cam = GradCAM(model=model, target_layers=target_layers)
                targets = [ClassifierOutputTarget(y_label)]

                grayscale_cam = cam(input_tensor=x, targets=targets)[0, :]
                # grayscale_cam = grayscale_cam[0, :]

                viz = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                ax = axes[prune_iteration]
                ax.imshow(viz)
                ax.set_title(f"Iter: {prune_iteration}\nPred: {classes[predicted]} ({'Correct' if predicted == y_label else 'Wrong'})", 
                            color=("green" if predicted == y_label else "red")
                            )
                ax.axis('off')

            plt.suptitle(f"Grad-CAM Across Pruning Iterations {exp.name} | Ground Truth: {classes[y_label]}", fontsize=16)
            plt.tight_layout()
            plt.show()
            break

        # break