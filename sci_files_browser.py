
# script to view datasets


import numpy as np
from numpy import ndarray
from pathlib import Path
import cv2 as cv
from matplotlib import pyplot as plt
from aicspylibczi import CziFile
import os

from config.config import DATASET_PATHS, DATASETS


# vopied from scene.py
def _get_img(sci_file) -> ndarray:
    img_block, dims_list = sci_file.read_image(S=0, Z=0)  
    img_squeezed = np.squeeze(img_block)
    return img_squeezed


if __name__ =="__main__":

    # location = DATASET_PATHS[DATASETS.YPD_Acetate_DAY_1_Acetate]["path"]
    from pathlib import Path

    # location = Path("C:/Work/Matfyz/Thesis/data/zeiss_matfyz/magMag 234.85/YPD_Acetate/Day 1/YPD1%")
    # location = Path("C:\Work\Matfyz\Thesis\data\zeiss_matfyz\magMag 234.85\YPD_SD_Acetate\Day 1\Acetate")
    # location = Path("C:\Work\Matfyz\Thesis\data\zeiss_matfyz\magMag 234.85\YPD_SD_Acetate\Day 1\SD")
    # location = Path("C:\Work\Matfyz\Thesis\data\zeiss_matfyz\magMag 234.85\YPD_SD_Acetate\Day 3\SD")
    location = Path("C:\Work\Matfyz\Thesis\data\zeiss_matfyz\magMag 234.85\YPD_SD_Acetate\Day 3\YPD")
    files = os.listdir(location)



    for file in files:
        print(file)

        # img_path = location.joinpath(file)
        # sci_file = CziFile(img_path)
        # img = _get_img(sci_file)

        # plt.imshow(img, cmap='gray')
        # plt.title(file)
        # plt.tight_layout()
        # plt.show()
