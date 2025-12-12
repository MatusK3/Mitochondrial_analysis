import skimage
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma
import cv2

from dataset_loader import Dataset
from config.config import DATASET_PATHS, DATASETS

from matplotlib import pyplot as plt



def get_roi(img, mask):
    ys, xs = np.where(mask)
    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()
    return img[ymin:ymax+1, xmin:xmax+1], mask[ymin:ymax+1, xmin:xmax+1]

def get_masked(img, mask):
    return np.where(mask, img, 0)

def normalize_01(img: np.ndarray, mask):
    img = img.astype(np.float32)
    masked_data = img[mask]

    # norm_img = (img - masked_data.mean()) / masked_data.std()
    norm_img = (img - masked_data.min()) / (masked_data.max() - masked_data.min())

    return norm_img

def denoise(gray_img):
    img_bgr = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    sigma_est = np.mean(estimate_sigma(img_bgr, channel_axis=-1))
    patch_kw = dict(
        patch_size=5,  # 5x5 patches
        patch_distance=6,  # 13x13 search area
        channel_axis=-1,
    )
    denoised_bgr = denoise_nl_means(img_bgr, h=0.8 * sigma_est, fast_mode=True, **patch_kw)

    return cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2GRAY)

def stadard_preprocess(img, mask):
    roi, mask_roi = get_roi(img, mask)
    roi = get_masked(roi, mask_roi)
    roi = normalize_01(roi, mask_roi)
    denoised_roi = denoise(roi)
    return roi, mask_roi, denoised_roi


if __name__ =="__main__":
    d = Dataset(DATASET_PATHS[DATASETS.YPD_SD_Acetate_DAY_3_Acetate])


    for scene in d.scenes:
        first_not_None_layer_id = next((i for i, x in enumerate(scene.dark) if x is not None), None)
        gray_img = scene.dark[first_not_None_layer_id]

        for i, mask in enumerate(scene.masks):
            fig, axes = plt.subplots(2, 2, figsize=(8, 8))

            axes[0, 0].imshow(gray_img, cmap='gray')
            axes[0, 0].set_title(f'dark, {scene.dark_names[first_not_None_layer_id]}')

            axes[0, 1].imshow(mask, cmap='gray')
            axes[0, 1].set_title(f'mask {i}/{len(scene.masks)}')


            roi, mask_roi = get_roi(gray_img, mask)
            roi = get_masked(roi, mask_roi)
            roi = normalize_01(roi, mask_roi)

            axes[1, 0].imshow(roi, cmap='gray')
            axes[1, 0].set_title(f'normd roi')


            denoised_img = denoise(roi)

            roi, mask_roi, denoised_img = stadard_preprocess(gray_img, mask)

            axes[1, 1].imshow(denoised_img, cmap='gray')
            axes[1, 1].set_title(f'denoised roi')

            plt.show()