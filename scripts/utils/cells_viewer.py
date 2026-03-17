





import numpy as np
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from aicspylibczi import CziFile
import cv2
from tqdm.auto import tqdm

def load_czi_img(sci_file_path) -> np.ndarray: # sci/czi file loader
    sci_file = CziFile(sci_file_path)
    img_block, dims_list = sci_file.read_image(S=0, Z=0)  
    img_squeezed = np.squeeze(img_block)
    return img_squeezed



DATASET_PATH = Path("data/mito")
config_path = Path.joinpath(DATASET_PATH, "config.json")


# def get_all(env):
#     with open(config_path, 'r') as f:
#         config = json.load(f)
#     dataset_structure = pd.read_csv(Path.joinpath(DATASET_PATH, config["scans_list_path"]))

#     dataset_structure_selected_env = dataset_structure[dataset_structure['environment'] == env]

#     all_scans = [
#         (
#             scans.loc[scans['light'] == 1, 'name'].iloc[0].replace(".czi", ""), # light img
#             scans.loc[scans['light'] == 0, 'name'].iloc[0].replace(".czi", "")  # dark img
#         )
#         for scene_id, scans in dataset_structure_selected_env.groupby("scene")
#     ]

#     return all_scans


def normalize_masked_region(image, mask):
    mask_bool = mask.astype(bool)
    roi_pixels = image[mask_bool]

    mean = np.mean(roi_pixels)
    std = np.std(roi_pixels)
    if std == 0:
        return np.zeros_like(image, dtype=np.float32)
    normalized = (image.astype(np.float32) - mean) / std
    normalized[~mask_bool] = normalized.max() # mask background

    return normalized



def get_all(mask_dir, env):
    with open(config_path, 'r') as f:
        config = json.load(f)
    dataset_structure = pd.read_csv(Path.joinpath(DATASET_PATH, config["scans_list_path"]))
    dataset_img_dir_path = Path.joinpath(DATASET_PATH, config["relative_img_path"], env)
   
    batch_rois = []
    batch_masks = []
    batch_spacing = []
    batch_metadata = []

    print(20*"==", env)
    print(dataset_structure[dataset_structure['environment'] == env])
    
    splitted_dataset_structure = dataset_structure[dataset_structure['environment'] == env]
    scene_pbar = tqdm(splitted_dataset_structure.groupby("scene"), desc=f"fetching {env} scenes", unit="scene")
    for scene_id, scans in scene_pbar:
        light_name = scans.loc[scans['light'] == 1, 'name'].iloc[0].replace(".czi", "")
        dark_name = scans.loc[scans['light'] == 0, 'name'].iloc[0].replace(".czi", "")
        # env = scans['environment'].iloc[0]
        
        # img_path = f"{img_dir}/{env}/{dark_name}.npy"
        img_path = f"{dataset_img_dir_path}/{dark_name}.czi"
        mask_path = f"{mask_dir}/{light_name}.png"

        # img = np.load(img_path)
        img = load_czi_img(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None: continue

        unique_ids = np.unique(mask)
        for uid in unique_ids:
            if uid == 0: continue # skip background
            
            obj_mask = (mask == uid).astype(np.uint8)
            if obj_mask.sum() < 1000:
                print("too small mask")
                continue

            y_indices, x_indices = np.where(obj_mask)
            if len(y_indices) == 0: 
                print("empty")
                continue # empty mask

            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()

            roi = img[y_min:y_max+1, x_min:x_max+1]
            roi_mask = obj_mask[y_min:y_max+1, x_min:x_max+1]



            batch_rois.append(normalize_masked_region(roi, roi_mask))
            # batch_spacing.append([voxel_spacing[0] * 1e8, voxel_spacing[1] * 1e8])
            batch_masks.append(roi_mask)
            batch_metadata.append({"img_name": dark_name, "scene_id": scene_id, "mask_index": uid})

    if len(batch_rois) != 0:
        return batch_rois, batch_masks, batch_spacing, batch_metadata


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Process dark/light image pairs.")
    
    # parser.add_argument("--segm_dir", required=True, help="path to dir, with segmentations")
    # parser.add_argument("--enviroment", required=True, help="path to dir with loaded czi images saved as numpy")
    # parser.add_argument("--output", required=True, help="Path to output csv file")

    # args = parser.parse_args()

    # extract_fetures(args.segm_dir, args.enviroment, args.output)

    DATASET_PATH = Path("data/mito")

    config_path = Path.joinpath(DATASET_PATH, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    


    for env in [
                "Acetate_DAY1",
                "Acetate_DAY3",
                "SD_DAY1",
                "SD_DAY3",
                "YPD_DAY1",
                "YPD_DAY3",
                "YPD",
                "YPD%1_DAY1",
                "YPD%1_DAY2",
                "YPD%1_DAY3",
                "YPGly",
                "YPGly_DAY1",
                "YPGly_DAY2",
                "YPGly_DAY3",
                "YPGal",
                "YPGal_DAY1",
                "YPGal_DAY2",
                "YPGal_DAY3"
                ]:
        segm_dir = f"results/segmentation/cellpose/{env}"
        dataset_img_dir_path = Path.joinpath(DATASET_PATH, config["relative_img_path"], env)
        batch_rois, batch_masks, batch_spacing, batch_metadata = get_all(segm_dir, env)


        # GRID WINDOW
        grid_dim = 5

        # Create a second figure object
        fig, axes = plt.subplots(grid_dim, grid_dim, figsize=(13, 13))
        # fig.suptitle(env, fontsize=16)
        axes = axes.flatten()

        rng = np.random.default_rng(seed=42)
        random_choices = rng.choice(len(batch_rois), size=grid_dim**2, replace=False)

        for i in range(grid_dim**2):
            # img_name = env_scans[i][1]
            # img_pth = f"{dataset_img_dir_path}/{img_name}.czi"
            # img = load_czi_img(img_pth)
            img = batch_rois[random_choices[i]]
            img_name = batch_metadata[random_choices[i]]["img_name"]
            mask_index = batch_metadata[random_choices[i]]["mask_index"]

            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"img: {img_name}; cell: {mask_index}", fontsize=8)
            axes[i].axis('off')

        # Hide empty subplots in the grid
        # for j in range(i + 1, len(axes2)):
            # axes2[j].axis('off')

        fig.tight_layout()

        plt.tight_layout()

        plt.savefig(f'results/cells_expl/{env}.png')
        # plt.show()


