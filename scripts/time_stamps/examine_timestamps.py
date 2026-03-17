# envs/czi_env.yaml  snakemake enviroment:
# conda activate .snakemake/conda/0aa85d447416ef846da9cedbe996cbc3_


import numpy as np
from aicspylibczi import CziFile
import pandas as pd
from pathlib import Path
import json
import re

def get_timestamp(sci_file_path: Path) -> str: # sci/czi file loader
    sci_file = CziFile(sci_file_path)

    xml_str = sci_file.meta
    acquisition_date_element = xml_str.find(".//AcquisitionDateAndTime")
    if acquisition_date_element is not None:
        timestamp = acquisition_date_element.text
        return timestamp
    return None

    # img_block, dims_list = sci_file.read_image(S=0, Z=0)  
    # img_squeezed = np.squeeze(img_block)

    # x_spacing = float(sci_file.meta.find("Metadata/Scaling/Items/Distance[@Id='X']/Value").text)
    # y_spacing = float(sci_file.meta.find("Metadata/Scaling/Items/Distance[@Id='Y']/Value").text)
    # voxel_spacing = [y_spacing, x_spacing] # meters to millimeters

    # return img_squeezed, voxel_spacing

from datetime import datetime
def get_min_max_median_ids(timestamps):
    dt_list = [
        datetime.fromisoformat(ts.replace('Z', '+00:00'))
        for ts in timestamps
    ]

    sorted_indices = sorted(range(len(dt_list)), key=lambda i: dt_list[i])

    min_index = sorted_indices[0]

    # Max index
    max_index = sorted_indices[-1]

    # Median index
    median_index = sorted_indices[len(sorted_indices) // 2]

    return min_index, max_index, median_index




DATASET_PATH = Path("data/mito")

config_path = Path.joinpath(DATASET_PATH, "config.json")
with open(config_path, 'r') as f:
    config = json.load(f)
dataset_structure = pd.read_csv(Path.joinpath(DATASET_PATH, config["scans_list_path"]))

dataset_img_dir_path = Path.joinpath(DATASET_PATH, config["relative_img_path"])

# SCENE_MAP = {
#     str(scene_id) : {
#         "light_name" : scans.loc[scans['light'] == 1, 'name'].iloc[0].replace(".czi", ""),
#         "dark_name" : scans.loc[scans['light'] == 0, 'name'].iloc[0].replace(".czi", ""),
#         "env" : scans['environment'].iloc[0]
#     }
#     for scene_id, scans in dataset_structure.groupby("scene")
# }

# str(dataset_img_dir_path) + "/{env}/{img}.czi"

envs = []
env_ts = []
env_names = []
for env, env_svcenes in dataset_structure.groupby("environment"):
    # env = env_svcenes["environment"].iloc[0]
    envs.append(env)
    env_ts.append([])
    env_names.append([])
    for scene_id, scene in env_svcenes.groupby("scene"):
        light_name = scene.loc[scene['light'] == 1, 'name'].iloc[0].replace(".czi", "")
        dark_name = scene.loc[scene['light'] == 0, 'name'].iloc[0].replace(".czi", "")

        light_ts = get_timestamp(Path(str(dataset_img_dir_path) + f"/{env}/{light_name}.czi"))
        dark_ts = get_timestamp(Path(str(dataset_img_dir_path) + f"/{env}/{dark_name}.czi"))
    
        env_ts[-1].append(light_ts)
        env_names[-1].append(light_name)

        env_ts[-1].append(dark_ts)
        env_names[-1].append(dark_name)


for env_id, env in enumerate(envs):
    print(env)
    min_index, max_index, median_index = get_min_max_median_ids(env_ts[env_id])

    print(f"min: {env_ts[env_id][min_index]};       name: {env_names[env_id][min_index]}")
    print(f"med: {env_ts[env_id][median_index]};       name: {env_names[env_id][median_index]}")
    print(f"max: {env_ts[env_id][max_index]};       name: {env_names[env_id][max_index]}")
    print("\n\n")