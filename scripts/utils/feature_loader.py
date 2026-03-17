
from typing import List, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import json
from enum import Enum

DATASET_PATH = Path("data/mito")
config_path = Path.joinpath(DATASET_PATH, "config.json")

class DATASETS_SPLITS(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

def load_features(data_dir: str, splits_list: list[DATASETS_SPLITS] = [DATASETS_SPLITS.TRAIN, DATASETS_SPLITS.VAL], features_preselction: list[str] =None) -> dict[str, dict[str, pd.DataFrame]]:#Tuple[pd.DataFrame,  pd.DataFrame]:
    # LOAD DATA
    with open(config_path, 'r') as f:
        config = json.load(f)
    classes = config["classes"]
    df_data = pd.concat([pd.read_csv(f"{data_dir}/{c}.csv") for c in classes], ignore_index=True)


    # create maping, that for given scene id maps its enviroment (used for labels)
    dataset_structure = pd.read_csv(Path.joinpath(DATASET_PATH, config["scans_list_path"]))
    env_map = dataset_structure.set_index('scene')['environment'].to_dict() # Create a mapping dictionary {scene: env}

    data = {}
    # SELECT ROWS/DATASET_SPLIT
    for split in splits_list:
        data[split] = {}
        split_scenes = config["dataset_split"][split.value]
        split_df_data = df_data[df_data['scene_id'].isin(split_scenes)]

        # FILTER COLUMNS/FEATURS
        if features_preselction is not None:
            # keep only selected set of features
            features = split_df_data[features_preselction]
        else:
            # filter out metadata, keep all features
            additional_methadata = ["sample_name", "scene_id", "mask_index"] # mirp sample_name + my additional columns
            features  = split_df_data.drop(columns=[c for c in df_data.columns if c.startswith("image_") or c in additional_methadata]) # all metadata from mirp.extract_feature() statrts wit "img"
        features = features.replace([np.inf, -np.inf], np.nan) # get rif of non compatible values
        features = features.dropna(axis=1)  # drop columns with any NaN

        labels = split_df_data["scene_id"].map(env_map).map(config["class_mapping"]).map(config["class_to_int_mapping"])

        data[split]["x"] = features
        data[split]["y"] = labels

    # labels = df_data["scene_id"].map(env_map).map(config["class_mapping"]).map(config["class_to_int_mapping"])

    return data


if __name__ == "__main__":
    # load_features("results/features/mirp_test.csv")
    ...