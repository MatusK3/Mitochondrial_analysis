from czi_img_loader import load_czi_img
import pandas as pd
import json
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import random


def get_classes_by_calss_id(class_mapping, class_to_int_mapping, class_id):
    return [
        key
        for key, class_name in class_mapping.items()
        if class_to_int_mapping.get(class_name) == class_id
    ]


def load_segm_instances(image_path):
    png_masks = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)    
    n_channels = 1 + np.max(png_masks)
    channels = np.eye(n_channels, dtype=bool)[png_masks]
    channels = np.rollaxis(channels, 2)
    return channels[1:]

if __name__ == "__main__":

    show_mode = [
        # "light_dark",
        # "dark",
        "light_segmentation"
    ]





    DATSET_PATH = "data/mito"
    csv_path = f"{DATSET_PATH}/mito.csv"
    config_path = f"{DATSET_PATH}/config.json"

    df = pd.read_csv(csv_path)
    with open(config_path, 'r') as f:
        config = json.load(f)

    class_mapping = config['class_mapping']
    class_to_int_mapping = config['class_to_int_mapping']


    img_path = f"{DATSET_PATH}/{config['relative_img_path']}"
    mask_path = f"{DATSET_PATH}/{config['relative_mak_path']}"

    for class_id in range(5):
        if class_id != 1: continue
        for scene_id, group in df.groupby("scene"):
            env = group["environment"].iloc[0]

            if env not in get_classes_by_calss_id(class_mapping, class_to_int_mapping, class_id):
                continue

            light_name = group.loc[group["light"] == 1, "name"].iloc[0]
            dark_name = group.loc[group["light"] == 0, "name"].iloc[0]
            
        
            light_path = f"{img_path}/{env}/{light_name}"
            dark_path = f"{img_path}/{env}/{dark_name}"

            # label
            # segmentaion_path = f"{mask_path}/{env}/{os.path.basename(light_name.split('.')[-2])}.png"
            # cellpose
            segmentaion_path = f"results/segmentation/cellpose/{env}/{os.path.basename(light_name.split('.')[-2])}.png"

            masks = load_segm_instances(segmentaion_path)

            light = load_czi_img(light_path)
            dark = load_czi_img(dark_path)

            if "light_dark" in show_mode:
                fig, axes = plt.subplots(1, 2, figsize=(16, 8))
                axes = axes.flatten()

                axes[0].imshow(light, cmap='gray')
                axes[0].set_title('light')
                axes[0].axis('off')

                axes[1].imshow(dark, cmap='gray')
                axes[1].set_title('dark')
                axes[1].axis('off')

                plt.suptitle(f"{class_mapping[env]}", fontsize=16)
                plt.tight_layout()
                plt.show()

            if "dark" in show_mode:
                fig = plt.figure(figsize=(10,8))

                plt.imshow(dark, cmap='gray')
                plt.axis("off")

                plt.suptitle(f"{class_mapping[env]}", fontsize=16)
                plt.tight_layout()
                plt.show()

            if "light_segmentation" in show_mode:
                fig, axes = plt.subplots(2, 1, figsize=(8, 14))
                axes = axes.flatten()
                axes[0].imshow(light, cmap='gray')
                axes[0].set_title('light')
                axes[0].axis('off')

                axes[1].imshow(light, cmap='gray')
                axes[1].set_title('cellpsoe')
                axes[1].axis('off')
                for i, mask in enumerate(masks):
                    color = np.array(mcolors.to_rgba(cm.tab20(random.randint(0, 19)), alpha=0.6))
                    overlay = np.zeros((*mask.shape, 4))
                    overlay[mask] = color
                    axes[1].imshow(overlay)
                    # plt.contour(mask, colors='k', linewidths=2)

                # axes[1,1].imshow(np.sum(self.masks, axis=0), cmap='gray')
                # axes[0,1].set_title('boolean masks')
                # axes[0,1].axis("on")


                plt.suptitle(f"{class_mapping[env]}", fontsize=16)
                plt.tight_layout()
                plt.show()

            break



    