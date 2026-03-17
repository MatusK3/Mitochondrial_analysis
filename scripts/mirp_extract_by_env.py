import argparse
import numpy as np
import cv2
import json
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

from czi_img_loader import load_czi_img

from mirp import extract_features
from mirp.settings.generic import SettingsClass
from mirp.settings.transformation_parameters import ImageTransformationSettingsClass
from mirp.settings.feature_parameters import FeatureExtractionSettingsClass
from mirp.settings.resegmentation_parameters import ResegmentationSettingsClass
from mirp.settings.perturbation_parameters import ImagePerturbationSettingsClass
from mirp.settings.image_processing_parameters import ImagePostProcessingClass
from mirp.settings.interpolation_parameters import ImageInterpolationSettingsClass, MaskInterpolationSettingsClass
from mirp.settings.general_parameters import GeneralSettingsClass




DATASET_PATH = Path("data/mito")
config_path = Path.joinpath(DATASET_PATH, "config.json")


def get_mirp_feature_extraction_settings() -> SettingsClass:
    discretisation_method="fixed_bin_number"
    discretisation_n_bins=32
    #new_spacing = [6] # to how many meter corresponds each pixel. As not each image has this ratio pixel/meter same, they are all resized to shared ratio (need to know orginal "voxel_spacing" of each scene)
    new_spacing = [6, 6]

    general_settings = GeneralSettingsClass(
        by_slice =True,
        ibsi_compliant =False # LBP are not in IBSI
    )
    post_processor = ImagePostProcessingClass(
        # no normalization, i will do it manually
        intensity_normalisation="none", # This transforms your intensity values so they have a mean (μ) of 0 and a standard deviation (σ) of 1.
        # intensity_normalisation_params={"roi_only": True},
        tissue_mask_type = "none"
    )
    
    image_interpolation_settings = ImageInterpolationSettingsClass(
        by_slice =general_settings.by_slice,
        new_spacing =new_spacing,  # different scenes, can have different pixel/reality scales. Use voxel_spacing and set all scenes to same scale
        spline_order = 3  # interpolation_method, cubic
    )

    mask_interpolation_settings = MaskInterpolationSettingsClass(
        # by_slice=general_settings.by_slice,
        # new_spacing=new_spacing,
        roi_spline_order = 0 # interpolation_method, nearest
    )
    # Feature extraction parameters
    feature_computation_parameters = FeatureExtractionSettingsClass(
        by_slice =general_settings.by_slice,
        ibsi_compliant=general_settings.ibsi_compliant,
        no_approximation=True,
        base_feature_families="all",  # compute all radiomics families
        base_discretisation_method =discretisation_method,
        base_discretisation_n_bins =discretisation_n_bins,
        stat_percentile = [10.0, 90.0],
        stat_value_shift = 0.0,
        ivh_discretisation_method = discretisation_method,
        # base_discretisation_bin_width=None,
        glcm_distance =[1.0, 2.0], # voxel/pixel distance
        glcm_spatial_method =["2d_average"],
        glrlm_spatial_method =["2d_average"],
        glszm_spatial_method =["2d"],
        gldzm_spatial_method =["2d"],
        ngtdm_spatial_method =["2d"],
        ngldm_distance =[1.0, 2.0], # voxel/pixel distance
        ngldm_spatial_method =["2d"],
        # ngldm_difference_level=[0.0]
    )
    image_transformation_settings = ImageTransformationSettingsClass(
        by_slice =general_settings.by_slice,
        ibsi_compliant =general_settings.ibsi_compliant,

        response_map_feature_families="all",
        response_map_discretisation_method =discretisation_method,
        response_map_discretisation_n_bins =discretisation_n_bins,

        # filter_kernels=["mean", "lbp"],
        # mean_filter_kernel_size = 3,
        filter_kernels=["lbp"],
        boundary_condition = "mirror", # scipy.ndimage.convolve
        lbp_method=["rotation_invariant"],
        lbp_filter_distance=[2],
    )
    settings = SettingsClass(
        general_settings=general_settings,
        post_process_settings=post_processor, #ImagePostProcessingClass(),
        img_interpolate_settings=image_interpolation_settings,
        roi_interpolate_settings=None,
        roi_resegment_settings=None,  # no resegmentation
        perturbation_settings=None,   # no perturbation
        img_transform_settings=image_transformation_settings,
        feature_extr_settings=feature_computation_parameters
    )
    return settings


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
        img, voxel_spacing = load_czi_img(img_path)
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
            batch_spacing.append([voxel_spacing[0] * 1e8, voxel_spacing[1] * 1e8])
            batch_masks.append(roi_mask)
            batch_metadata.append({"scene_id": scene_id, "mask_index": uid})

    if len(batch_rois) != 0:
        return batch_rois, batch_masks, batch_spacing, batch_metadata





def normalize_masked_region(image, mask):
    mask_bool = mask.astype(bool)
    roi_pixels = image[mask_bool]

    mean = np.mean(roi_pixels)
    std = np.std(roi_pixels)
    if std == 0:
        return np.zeros_like(image, dtype=np.float32)
    normalized = (image.astype(np.float32) - mean) / std
    normalized[~mask_bool] = 0.0 # mask background

    return normalized






def extract_fetures(mask_dir, env, outpu_pth):

    settings = get_mirp_feature_extraction_settings()
    # features_dataset = []

    batch_rois_normalized, batch_masks, batch_spacing, batch_metadata = get_all(mask_dir, env) # fetch whole enviroment

    # for i in range(len(batch_rois_normalized)):
    #     print(batch_rois_normalized[i].shape, batch_masks[i].shape, batch_metadata[i])
    #     # if batch_metadata[i]["scene_id"] == "310" and  batch_metadata[i]["mask_index"] == 7:#{'scene_id': 310, 'mask_index': np.uint8(7)}
    #     batch_results = extract_features(
    #         image=batch_rois_normalized[i],
    #         mask=batch_masks[i],
    #         # intensity_normalisation="standardisation",
    #         settings=settings,
    #         voxel_spacing=batch_spacing[i],
    #         num_cpus=6,
    #         parallel_backend="joblib"
    #     )

    batch_results = extract_features(
        image=batch_rois_normalized,
        mask=batch_masks,
        # intensity_normalisation="standardisation",
        settings=settings,
        voxel_spacing=batch_spacing,
        num_cpus=6,
        parallel_backend="joblib"
    )
    # batch_results = [pd.DataFrame({"spacing_x": [bs[0]], "spacing_y": [bs[1]]}) for bs in batch_spacing] 

    # Attach metadata back to each result
    for i, df in enumerate(batch_results):
        df["scene_id"] = batch_metadata[i]["scene_id"]
        df["mask_index"] = batch_metadata[i]["mask_index"]
        # features_dataset.append(df)

    result = pd.concat(batch_results, ignore_index=True)
    result.to_csv(outpu_pth, index=False)
    print(f"FINISHED, saved to: {outpu_pth}")


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
        out_dir = f"results/features/mirp/{env}.csv"
        extract_fetures(segm_dir, env, out_dir)
