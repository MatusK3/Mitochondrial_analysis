
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from cellpose import models #, plot
import cv2 as cv
from matplotlib import pyplot as plt
from pathlib import Path


data_path = Path("D:/Olympus_magMag234.85_2025-02-26/DiOC6")
print()


file_paths = sorted(file_path for file_path in data_path.iterdir()
        if
            file_path.suffix == ".tif" and
            "scale" not in file_path.name and
            file_path.is_file()
        )

id = 1
img = cv.imread(str(file_paths[id]), 0)






# 'cyto' for whole-cell
# 'nuclei' for nucleus segmentation
model = models.Cellpose(model_type='cyto')

masks, flows, styles, diams = model.eval(
        img, 
        channels=[0,0], 
        normalize=True,
        diameter=600, # target cell diameter
        flow_threshold=0.2, # Lower values (e.g., 0.2) help detect larger objects by allowing looser constraints on flow consistency.
        cellprob_threshold=0.4, # Increase this to favor larger objects and remove small false positives.
    )


fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Original image
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original Image')

# Segmentation mask overlay
# mask_overlay = plot.mask_overlay(image, masks)
axes[1].imshow(masks, cmap='gray')
axes[1].set_title('Segmentation Mask')
axes[1].axis('off')

plt.tight_layout()
plt.show()