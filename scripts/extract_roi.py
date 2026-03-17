import numpy as np
import cv2
import argparse
import h5py


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read image and save it as numpy")
    parser.add_argument("--image", required=True, help="Path to input image, numpy format")
    parser.add_argument("--mask", required=True, help="Path to mask corresponding to image, png format")
    parser.add_argument("--output", required=True, help="Path to output, h5py format")
    args = parser.parse_args()


    img = np.load(args.image)
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)

    unique_ids = np.unique(mask)

    with h5py.File(args.output, 'w') as f:
        for uid in unique_ids:
            if uid == 0: continue

            obj_mask = (mask == uid).astype(np.uint8)

            y_indices, x_indices = np.where(obj_mask)
            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()

            roi_crop = img[y_min:y_max+1, x_min:x_max+1]
            mask_crop = obj_mask[y_min:y_max+1, x_min:x_max+1]
            roi = roi_crop * mask_crop

            f.create_dataset(str(uid), data=roi, compression="gzip")


