import pandas as pd
import json
from sklearn.model_selection import train_test_split

def split_dataset_by_scene(csv_path, json_path, train_pct=0.6, val_pct=0.2, test_pct=0.2, slack_factor=0.5, random_seed=42):
    # load_dataset_config
    with open(json_path, 'r') as f:
        config = json.load(f)

    class_mapping = config["class_mapping"]

    # load dataset structure
    df = pd.read_csv(csv_path)


    # temp filter classes:
    #  "class_mapping": {
    #     "Acetate_DAY1": "Neutral",
    #     "Acetate_DAY3": "Neutral",
    #     "SD_DAY1": "Neutral",
    #     "SD_DAY3": "SD_D3",
    #     "YPD_DAY1": "Neutral",
    #     "YPD_DAY3": "YPD_D3",
    #     "YPD": "Neutral",
    #     "YPD%1_DAY1": "Neutral",
    #     "YPD%1_DAY2": "YPD_D2",
    #     "YPD%1_DAY3": "YPD_D3",
    #     "YPGly": "Neutral",
    #     "YPGly_DAY1": "Neutral",
    #     "YPGly_DAY2": "YPGly_D2",
    #     "YPGly_DAY3": "YPGly_D3",
    #     "YPGal": "Neutral",
    #     "YPGal_DAY1": "Neutral",
    #     "YPGal_DAY2": "YPGal_D2",
    #     "YPGal_DAY3": "YPGal_D3"
    #     }

    df = df[df["environment"].isin(config["classes"])]
    # df = df[~df["environment"].isin(["YPD%1_DAY2", "YPGly_DAY2", "YPGal_DAY2"])]
    # df = df[df["environment"].isin(["YPD%1_DAY1", "YPD%1_DAY2", "YPD%1_DAY3", "YPGly_DAY1", "YPGly_DAY2", "YPGly_DAY3", "YPGal_DAY1", "YPGal_DAY2", "YPGal_DAY3"])]
    # df = df[df["environment"].isin(["Acetate_DAY1", "Acetate_DAY3", "SD_DAY1", "SD_DAY3", "YPD_DAY1", "YPD_DAY3"])]
    # df = df[df["environment"].isin(["YPD%1_DAY1", "YPD%1_DAY3", "YPGly_DAY1", "YPGly_DAY3", "YPGal_DAY1", "YPGal_DAY3"])]
    
    
    meta = df[['scene', 'environment']].drop_duplicates() # each scane has 2 entries => drop duplicate
    meta['logical_group'] = meta['environment'].map(class_mapping) # applay mapping

    # # downsampling, ballnaced, even if some classes are equivalent and groped into bigger logical classes, choose same ammout of class from each original class:
    # # from each subclass randomly select: min_group_size/num_of_sub_class
    class_distr = meta['logical_group'].value_counts()
    print(class_distr)
    config["dataset_class_distribution"] = class_distr.to_dict()

    if slack_factor is not None:
        min_group_size = class_distr.min()
        target_grope_size = min_group_size * (1+slack_factor) # slack_factor enables for lager classes use more of they resorces

        balanced_list = []
        for group_name, group_data in meta.groupby('logical_group'): # from each env select (+-)same size of samples
            sub_envs = group_data['environment'].unique()
            n_per_sub = int(target_grope_size / len(sub_envs))
            
            for env in sub_envs:
                env_subset = group_data[group_data['environment'] == env]
                n_to_sample = min(len(env_subset), n_per_sub) # target grope is larger than whole, sample smaller amonut.
                balanced_list.append(env_subset.sample(n=n_to_sample, random_state=random_seed))
        balanced_meta = pd.concat(balanced_list)

        print(f"Original scenes: {len(meta)}")
        print(f"Balanced scenes: {len(balanced_meta)} ({min_group_size} per logical group)")

        dataset_class_presence = balanced_meta['logical_group'].value_counts()
        # config["downsample_dataset_class_presence"] = dataset_class_presence.to_dict()
        print(f"class intensities: {dataset_class_presence}")
        meta = balanced_meta


    
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

    # collect unused scenes:
    # used_scenes = set(train_scenes) | set(val_scenes) | set(test_scenes)
    # unused_scenes = meta[~meta['scene'].isin(used_scenes)]['scene'].tolist()
    
    # final split dictionary
    split_dict = {
        'train': sorted([int(s) for s in train_scenes]),
        'val': sorted([int(s) for s in val_scenes]),
        'test': sorted([int(s) for s in test_scenes]),
        # 'unused': sorted([int(s) for s in unused_scenes]),
    }
    config['dataset_split'] = split_dict

    split_class_distribution = {
        name: meta[meta['scene'].isin(split)]['logical_group'].value_counts().to_dict()
        for split, name in [(train_scenes, "train"), (val_scenes, "val"), (test_scenes, "test")]
    }
    config["split_class_distribution"] = split_class_distribution
    # sanity check. check if all splits are exclusive:
    total_count = len(train_scenes) + len(val_scenes) + len(test_scenes) #+ len(unused_scenes)
    unique_count = len(set(train_scenes) | set(val_scenes) | set(test_scenes) )#| set(unused_scenes))
    if total_count != unique_count:
        raise ValueError("ERROR: Dataset has overlapping scenes between splits!")
    
    # add split to existing config.json
    # Update the key (stored as a list containing the dict as per your style)
    
    
    # Save back to JSON with nice indentation
    with open(json_path, 'w') as f:
        json.dump(config, f, indent=1)
    
    print("Done")
    print(f"{len(meta)} scenes split into: {len(train_scenes)} Train, {len(val_scenes)} Val, {len(test_scenes)} Test")#, {len(unused_scenes)} Unused.")
    print(f"total split size comparison: orignal:{len(meta)}; total_splitted:{total_count}")

if __name__ == "__main__":
    dataset_path = "data/mito/"
    data_structure_path = f"{dataset_path}/mito.csv"
    dataset_config_path = f"{dataset_path}/config.json"
    split_dataset_by_scene(data_structure_path, dataset_config_path, slack_factor=None) # slack_factor=0.5