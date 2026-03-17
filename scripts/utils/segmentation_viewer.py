import numpy as np
from aicspylibczi import CziFile
import matplotlib.pyplot as plt
import cv2
import h5py
import math
from pathlib import Path
import json
import pandas as pd

def load_czi_img(sci_file_path) -> np.ndarray: # sci/czi file loader
    sci_file = CziFile(sci_file_path)
    img_block, dims_list = sci_file.read_image(S=0, Z=0)  
    img_squeezed = np.squeeze(img_block)
    return img_squeezed


def on_key(event):
    if event.key == 'q':        # choose your key here
        plt.close('all')        # closes all open pyplot windows


DATASET_PATH = Path("data/mito")



config_path = Path.joinpath(DATASET_PATH, "config.json")
with open(config_path, 'r') as f:
    config = json.load(f)
dataset_structure = pd.read_csv(Path.joinpath(DATASET_PATH, config["scans_list_path"]))
dataset_img_dir_path = Path.joinpath(DATASET_PATH, config["relative_img_path"])


for scene_id in dataset_structure['scene'].unique():

    scene = dataset_structure.loc[dataset_structure['scene'] == scene_id]

    light_name = scene.loc[scene['light'] == 1, 'name'].iloc[0].replace(".czi", "")
    dark_name = scene.loc[scene['light'] == 0, 'name'].iloc[0].replace(".czi", "")
    env = scene['environment'].iloc[0]

    # masks_dir_path = f"results/segmentation/{classes[0]}"
    # image_dir_path = f"data/mito/img/{classes[0]}"
    # roi_dir_path = "results/roi"
    # img_name = "Snap-9128"
    # scene_id="759"



    mask_path = f"results/segmentation/cellpose/{env}/{light_name}.png"
    img_path = f"{dataset_img_dir_path}/{env}/{light_name}.czi"
    roi_path = f"results/roi/{scene_id}.h5"


    img = load_czi_img(img_path)
    masks = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    rois = []
    keys = []
    with h5py.File(roi_path, 'r') as f:
        for key in f.keys():
            keys.append(key)
            rois.append(f[key][()])

    # WINDOW 1: image and mask plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig.canvas.mpl_connect('key_press_event', on_key)

    ax[0].imshow(img, cmap='gray')
    ax[0].set_title(f"{env} {scene_id}")
    # ax[0].axis('off')

    # Use 'nipy_spectral' or 'prism' for high-contrast random colors for masks
    ax[1].imshow(masks, cmap='nipy_spectral')
    ax[1].set_title("Cellpose Masks")
    # ax[1].axis('off')

    # WINDOW 2: ROI Grid
    num_rois = len(rois)
    cols = 2
    rows = math.ceil(num_rois / cols)

    # Create a second figure object
    fig2, axes2 = plt.subplots(rows, cols, figsize=(15, rows * 3))
    fig2.suptitle("Extracted ROIs", fontsize=16)
    fig2.canvas.mpl_connect('key_press_event', on_key)
    axes2 = axes2.flatten()

    for i in range(num_rois):
        axes2[i].imshow(rois[i], cmap='gray')
        axes2[i].set_title(f"Key: {keys[i]}", fontsize=8)
        # axes2[i].axis('off')

    # Hide empty subplots in the grid
    # for j in range(i + 1, len(axes2)):
        # axes2[j].axis('off')

    fig2.tight_layout()

    plt.tight_layout()
    plt.show()