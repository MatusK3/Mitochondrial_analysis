# Featurewiz snakemake enviroment:
# conda activate .snakemake/conda/089e14863a2542433d6154648bf0f73e_

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from featurewiz import FeatureWiz


if __name__ == "__main__":
    random_seed = 42

    train_pct = 0.6
    val_pct = 0.2
    test_pct = 0.2

    featres_dir = "results/features/mirp"

    dataset_path = "data/mito/"
    data_structure_path = f"{dataset_path}/mito.csv"
    df = pd.read_csv(data_structure_path)

    env_config_pth = "scripts/env_experiments/config.json"
    # env_config_pth = "scripts/env_experiments/config_no_day_2.json"
    # env_config_pth = "scripts/env_experiments/config_ideal.json"
    with open(env_config_pth, 'r') as f:
        env_config = json.load(f)
    envs = env_config["envs"]

    output_json_pth = "scripts/env_experiments/split.json"
    output_json = {}

    output_json = {}
    for env in envs:
        mapping = env_config[env]
        output_json[env] = {}
        otp_split = output_json[env] 

        print("="*20 + "\n" + env)
        classes = mapping["classes"]
        env_df = df[df["environment"].isin(classes)]

        class_mapping = mapping["class_mapping"]
        meta = env_df[['scene', 'environment']].drop_duplicates() # each scane has 2 entries => drop duplicate
        meta['logical_group'] = meta['environment'].map(class_mapping) # applay mapping


        class_distr = meta['logical_group'].value_counts()
        print(class_distr)
        otp_split["class_distr"] = class_distr.to_dict()

        # train (Val + Test) split
        train_scenes, temp_scenes = train_test_split(
            meta['scene'],#balanced_meta['scene'],
            train_size=train_pct,
            random_state=random_seed,
            stratify=meta['logical_group']#balanced_meta['logical_group']
        )
    
        # Val + Test split
        relative_val_size = val_pct / (val_pct + test_pct) # relative split sizes
        # temp_balanced_meta = balanced_meta[balanced_meta['scene'].isin(temp_scenes)] # Extract environment labels for the temporary scenes to maintain stratification
        temp_meta = meta[meta['scene'].isin(temp_scenes)] # Extract environment labels for the temporary scenes to maintain stratification
        val_scenes, test_scenes = train_test_split(
            temp_scenes,
            train_size=relative_val_size,
            random_state=random_seed,
            stratify=temp_meta['logical_group']
        )

    
        # final split dictionary
        split_dict = {
            'train': sorted([int(s) for s in train_scenes]),
            'val': sorted([int(s) for s in val_scenes]),
            'test': sorted([int(s) for s in test_scenes]),
        }
        otp_split['split_scenes'] = split_dict

        split_class_distribution = {
            name: meta[meta['scene'].isin(split)]['logical_group'].value_counts().to_dict()
            for split, name in [(train_scenes, "train"), (val_scenes, "val"), (test_scenes, "test")]
        }
        otp_split["split_class_dist"] = split_class_distribution
        
        # sanity check. check if all splits are exclusive:
        total_count = len(train_scenes) + len(val_scenes) + len(test_scenes) #+ len(unused_scenes)
        unique_count = len(set(train_scenes) | set(val_scenes) | set(test_scenes) )#| set(unused_scenes))
        if total_count != unique_count:
            raise ValueError("ERROR: Dataset has overlapping scenes between splits!")


        # ==========================================================================================
        # Featurewiz

        # load features:
        target_scenes = otp_split['split_scenes']["train"]
        scne_id_class_mapping = meta.set_index('scene')['logical_group'].to_dict()

        df_data = pd.concat([pd.read_csv(f"{featres_dir}/{c}.csv") for c in classes], ignore_index=True)
        scenes_data = df_data[df_data['scene_id'].isin(target_scenes)]

        additional_methadata = ["sample_name", "scene_id", "mask_index"] # mirp sample_name + my additional columns
        features  = scenes_data.drop(columns=[c for c in df_data.columns if c.startswith("image_") or c in additional_methadata]) # all metadata from mirp.extract_feature() statrts wit "img"
        features = features.replace([np.inf, -np.inf], np.nan) # get rif of non compatible values
        features = features.dropna(axis=1)  # drop columns with any NaN

        class_to_int_mapping = mapping["class_to_int_mapping"]
        labels = scenes_data["scene_id"].map(scne_id_class_mapping).map(class_to_int_mapping)


        # featurewiz
        f_wiz = FeatureWiz(
            feature_engg='',
            corr_limit=0.9,  
            nrows=None,
            transform_target=False,
            category_encoders="auto",
            verbose=0 #2
        )
        x_train_selected, y_train_selected = f_wiz.fit_transform(features, labels)
        selected_features = f_wiz.features

        otp_split["filtered_features"] = selected_features

        print(f"num of features before: {features.shape[1]}; after: {len(selected_features)}")

    
    # Save back to JSON with nice indentation
    with open(output_json_pth, 'w') as f:
        json.dump(output_json, f, indent=1)
    
    print("Done")

    

    