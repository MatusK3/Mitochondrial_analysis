import os
import numpy as np
import cv2


path = "C:/Work/Matfyz/Thesis/data/zeiss_matfyz/magMag 234.85/annotations/YPD_SD_Acetate_old/DAY_1/SD"
save_path = "C:/Work/Matfyz/Thesis/data/zeiss_matfyz/magMag 234.85/annotations/YPD_SD_Acetate/DAY_1/SD"

for file in os.listdir(path):
    name = file.split(".")[0]
    file_path = f"{path}/{file}"
    masks = np.load(file_path, allow_pickle=True)
    
    img = np.zeros

    num_masks, img_x, img_y = masks.shape
    combined = np.zeros((img_x, img_y), dtype=np.uint8)
    
    for i in range(num_masks):
        mask_id = i + 1  # IDs from 1 to 255
        combined[masks[i]] = mask_id
    
    save_file_path = f"{save_path}/{name}.png"
    # print(save_file_path)
    # cv2.imwrite(save_file_path, combined)


    # png_masks = cv2.imread(save_file_path, cv2.IMREAD_GRAYSCALE)
    # n_channels = 1 + np.max(png_masks)
    # loaded_masks = np.eye(n_channels, dtype=bool)[png_masks]
    # print(masks.shape, loaded_masks.shape, np.rollaxis(loaded_masks, 2).shape, np.all(np.rollaxis(loaded_masks, 2)[1:] == masks))