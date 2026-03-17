# snakemake ml_vision conda env
# conda activate .snakemake/conda/7f69f82cf5e2d5fc9f5eb4294544276a_

import json
from pathlib import Path
import pandas as pd
import numpy as np

from itertools import combinations
from tqdm.auto import tqdm

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report



DATASET_PATH = Path("data/mito")
featres_dir = "results/features/mirp"

def load_data(scenes, classes, feature_names, class_mapping, class_to_int_mapping):
    dataset_structure = pd.read_csv(f"{DATASET_PATH}/mito.csv")
    env_map = dataset_structure.set_index('scene')['environment'].to_dict() # Create a mapping dictionary {scene: env}

    df_data = pd.concat([pd.read_csv(f"{featres_dir}/{c}.csv") for c in classes], ignore_index=True)
    scenes_data = df_data[df_data['scene_id'].isin(scenes)]

    features = scenes_data[feature_names]

    labels = scenes_data["scene_id"].map(env_map).map(class_mapping).map(class_to_int_mapping)

    return features, labels











if __name__ == "__main__":
    num_of_features_to_select = 2

    output_mean = f"scripts/env_experiments/results/top_{num_of_features_to_select}_mean.csv"
    output_min = f"scripts/env_experiments/results/top_{num_of_features_to_select}_min.csv"


    env_config_pth = "scripts/env_experiments/config.json"
    with open(env_config_pth, 'r') as f:
        env_config = json.load(f)
    envs = env_config["envs"]

    splits_config_pth = "scripts/env_experiments/split.json"
    with open(splits_config_pth, 'r') as f:
        splits_config = json.load(f)

    feature_names = list(set.union(*[set(splits_config[env]["filtered_features"]) for env in envs]))
    # i = set.intersection(*[set(splits_config[env]["filtered_features"]) for env in envs])

    data_collection = {}
    envs_pbar = tqdm(envs, desc=f"iterating enviroments", unit="env")
    for env in envs_pbar:
        data_collection[env] = {}
        scenes = np.concatenate([
            splits_config[env]["split_scenes"]["train"],
            splits_config[env]["split_scenes"]["val"]
        ])
        
        x, y = load_data(scenes, env_config[env]["classes"], feature_names, env_config[env]["class_mapping"], env_config[env]["class_to_int_mapping"])
        y = y.to_numpy()

        

        column_combinations = list(combinations(feature_names, num_of_features_to_select))
        column_combinations_pbar = tqdm(column_combinations, desc=f"iterating feature combination: {num_of_features_to_select} out of {len(feature_names)}", unit="comb")
        for feature_combination in column_combinations_pbar:
            feature_combination = list(feature_combination)
            feature_subse_x = x[feature_combination].to_numpy()

            rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=8, random_state=42)
            knn = KNeighborsClassifier(n_neighbors=5, weights='uniform') # uniform should help a little for imbalanced dataset
            scores = cross_val_score(knn, feature_subse_x, y, cv=rskf, scoring='f1_macro', n_jobs=6) # f1_macro for imbalanced datasets
            mean_f1 = scores.mean()
            std_f1 = scores.std()

            data_collection[env][" ".join(feature_combination)]= {
                "mean_f1_macro": mean_f1,
                "std_f1_macro": std_f1
            }


    total_score = []
    column_combinations_pbar = tqdm(data_collection[env].keys(), desc=f"iterating data computing total scores", unit="feature_comb")
    for feature_combination in column_combinations_pbar:
        scores = np.array([data_collection[env][feature_combination]["mean_f1_macro"] for env in envs])
        total_score.append({
            "feature_combination": feature_combination,
            "mean_f1": scores.mean(),
            'min_f1': scores.min(),
            **{
                f"{env}_mean_f1_macro": data_collection[env][feature_combination]["mean_f1_macro"] for env in envs
            }
        })


    df = pd.DataFrame(total_score)
    TOP_N = 50

    df_sorted = df.sort_values(by='mean_f1', ascending=False)
    top = df_sorted.head(TOP_N)
    top.to_csv(output_mean, index=False)

    df_sorted = df.sort_values(by='min_f1', ascending=False)
    top = df_sorted.head(TOP_N)
    top.to_csv(output_min, index=False)
