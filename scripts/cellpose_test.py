import numpy as np
from pathlib import Path
import json

import cv2
from cellpose import models, core, io, plot
# from tqdm import trange
import matplotlib.pyplot as plt
# from natsort import natsorted

# from skimage.io import imread, imsave

from utils.czi_scene_loader import Czi_scene_dataset, load_czi_img



config_path = Path("config.json")
with open(config_path, 'r') as f:
    config = json.load(f)



# io.logger_setup() # run this to get printing of progress

#Check if GPU access
if core.use_gpu()==False:
    raise ImportError("No GPU access, change your runtime")

model = models.CellposeModel(gpu=True)

dataset_path = Path(config["dataset_path"])
dataset = Czi_scene_dataset(dataset_path)

import random
# random.shuffle(dataset.scenes)
for scene in dataset.scenes:
    if scene.scene_id not in {0,4,100,105,162,170}: continue
    img = load_czi_img(scene.light_scan_path)
    img_name = scene.light_scan_name

    print(img.dtype)

    
    # k = 11#17
    # kernel = np.ones((k,k),np.float32)/k**2
    # img = cv2.filter2D(img,-1,kernel)


# image_path = 'L_100X/L_100X_C2/images/train/38_2_1000_ALL.png'
# image = np.asarray(Image.open(image_path))

    # parameter description: https://cellpose.readthedocs.io/en/latest/settings.html
    masks, flows, styles = model.eval(
        img,
        diameter=210,
        flow_threshold=0.5,
        cellprob_threshold=0.5,
        channels=[0, 0]
    )
    masks = np.astype(masks, np.uint8)
    print(masks.shape)
    print(masks.dtype)


    # cv2.imwrite(Path.joinpath(config["segmentation_dir"], img_name), masks)



    # plt.figure(figsize=(8, 8))
    # plt.imshow(masks, cmap="nipy_spectral")
    # plt.axis("off")
    # plt.title("img_name")

    # plt.tight_layout()
    # plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(img, cmap='gray')
    ax[0].set_title(img_name + "; " + scene.environment)
    ax[0].axis('off')

    # Use 'nipy_spectral' or 'prism' for high-contrast random colors for masks
    ax[1].imshow(masks, cmap='nipy_spectral')
    ax[1].set_title("Cellpose Masks")
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()