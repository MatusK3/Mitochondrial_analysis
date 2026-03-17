# snakemake ml_vision conda env
# conda activate .snakemake/conda/7f69f82cf5e2d5fc9f5eb4294544276a_

import json
from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay #confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, cross_val_predict





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

def evaluate_knn(train_x, train_y, test_x, test_y, target_names):
    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    knn.fit(train_x, train_y)
    y_pred = knn.predict(test_x)

    unique_y = set([*y_pred, *test_y])
    print(len(unique_y), ";", unique_y)
    filtered_tragts = {name: i for name, i in target_names.items() if i in y_pred or i in test_y}
    print(classification_report(test_y, y_pred, target_names=filtered_tragts))
    return f1_score(test_y, y_pred, average="macro")







def find_best_texture(train_x, train_y, val_x, val_y, feature_names):
    data_collection = []
    for feature in feature_names:
        feature_subse_train_x = train_x[[feature]].to_numpy()
        feature_subse_val_x = val_x[[feature]].to_numpy()

        X = np.concatenate([feature_subse_train_x, feature_subse_val_x])
        y = np.concatenate([train_y, val_y])

        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=8, random_state=42)

        knn = KNeighborsClassifier(n_neighbors=5, weights='uniform') # uniform should help a little for imbalanced dataset
        # f1_macro; f1_weighted; f1_micro
        scores = cross_val_score(knn, X, y, cv=rskf, scoring='f1_macro', n_jobs=5) # f1_macro for imbalanced datasets

        mean_f1 = scores.mean()
        std_f1 = scores.std()
        # print(f"Mean f1_macro: {mean_f1:.4f} (std f1_macro: {std_f1})")

        data_collection.append({
            "feature": feature,
            "mean_f1_macro": mean_f1,
            "std_f1_macro": std_f1
        })

    df = pd.DataFrame(data_collection)
    best_exp = df.sort_values(by=['mean_f1_macro', 'std_f1_macro'], ascending=[False, True]).iloc[0]
    return best_exp['feature'], best_exp['mean_f1_macro']






if __name__ == "__main__":
    # env_config_pth = "scripts/env_experiments/config.json"
    # env_config_pth = "scripts/env_experiments/config_no_day_2.json"
    env_config_pth = "scripts/env_experiments/config_ideal.json"
    with open(env_config_pth, 'r') as f:    
        env_config = json.load(f)
    envs = env_config["envs"]

    splits_config_pth = "scripts/env_experiments/split.json"
    with open(splits_config_pth, 'r') as f:
        splits_config = json.load(f)

    
    env_dims = ["YPD", "YPGly", "YPGal"]
    dims_features = []
    for env in env_dims:
        tarin_scenes = splits_config[env]["split_scenes"]["train"]
        val_scenes = splits_config[env]["split_scenes"]["val"]
        test_scenes = splits_config[env]["split_scenes"]["test"]


        classes = env_config[env]["classes"]
        preselected_features = splits_config[env]["filtered_features"]
        class_mapping = env_config[env]["class_mapping"]
        class_to_int_mapping = env_config[env]["class_to_int_mapping"]
        
        int_to_class_mapping = {v: k for k, v in class_to_int_mapping.items()}


        x_train, y_train = load_data(tarin_scenes, classes, preselected_features, class_mapping, class_to_int_mapping)
        x_val, y_val = load_data(val_scenes, classes, preselected_features, class_mapping, class_to_int_mapping)
        test_x, test_y = load_data(test_scenes, classes, preselected_features, class_mapping, class_to_int_mapping)
        y_train, y_val, test_y = y_train.to_numpy(), y_val.to_numpy(), test_y.to_numpy()

        feature, val_f1 = find_best_texture(x_train, y_train, x_val, y_val, preselected_features)
        dims_features.append(feature)

        X_embedded = np.concatenate([x_train[[feature]].to_numpy(), x_val[[feature]].to_numpy()])
        y = np.concatenate([y_train, y_val])
        test_x_embeded = test_x[[feature]].to_numpy()

        print(env)
        f1 = evaluate_knn(X_embedded, y, test_x_embeded, test_y, class_to_int_mapping)

        plt.figure(figsize=(8, 8))
        for class_int in np.unique(y):
            mask = y == class_int
            plt.hist(X_embedded[mask], label=int_to_class_mapping[class_int], alpha=0.5, bins=30)

        plt.title(f"{env}; val: {val_f1}; test: {f1}")
        plt.xlabel(feature)
        plt.legend()




        
    # all
    scenes = np.concatenate([
            np.concatenate([
                splits_config[env]["split_scenes"]["train"],
                splits_config[env]["split_scenes"]["val"]
            ])for env in env_dims
        ])
    test_scenes = np.concatenate([
        splits_config[env]["split_scenes"]["test"] for env in env_dims
    ])
    
    all_classes = [cls for env in env_dims for cls in env_config[env]["classes"]]
    # preselected_features = splits_config[env]["filtered_features"]
    all_class_mapping = {key: val for env in env_dims for key, val in env_config[env]["class_mapping"].items()}
    all_class_to_int_mapping = env_config["class_to_int_mapping"]
    
    feature_names = list(set.union(*[set(splits_config[env]["filtered_features"]) for env in envs]))
    x, y = load_data(scenes, all_classes, feature_names, all_class_mapping, all_class_to_int_mapping)
    test_x, test_y = load_data(test_scenes, all_classes, feature_names, all_class_mapping, all_class_to_int_mapping)
    y, test_y = y.to_numpy(), test_y.to_numpy()


    embedded_x = np.hstack([x[[feature]].to_numpy() for feature in dims_features])
    embedded_test_x = np.hstack([test_x[[feature]].to_numpy() for feature in dims_features])

    print("all eval")
    f1 = evaluate_knn(embedded_x, y, embedded_test_x, test_y, all_class_to_int_mapping)
    

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.concatenate([embedded_x, embedded_test_x])
    y = np.concatenate([y, test_y])
    filtered_tragts = {name: i for name, i in all_class_to_int_mapping.items() if i in y}
    for i, (cls_name, cls_id) in enumerate(filtered_tragts.items()):
        class_mask = y == cls_id
        ax.scatter(
            x[class_mask][:, 0],
            x[class_mask][:, 1],
            x[class_mask][:, 2],
            label=f'{cls_name}',
            s=50
        )
    
    ax.view_init(elev=15, azim=-125) # Set camera
    ax.set_xlabel(f"{env_dims[0]} - {dims_features[0]}")
    ax.set_ylabel(f"{env_dims[1]} - {dims_features[1]}")
    ax.set_zlabel(f"{env_dims[2]} - {dims_features[2]}")
    ax.legend()
    # plt.show()
        



    scenes = np.concatenate([
            np.concatenate([
                splits_config[env]["split_scenes"]["train"],
                splits_config[env]["split_scenes"]["val"]
            ])for env in env_config["envs"]
        ])
    test_scenes = np.concatenate([
        splits_config[env]["split_scenes"]["test"] for env in env_config["envs"]
    ])
    all_classes = [cls for env in env_config["envs"] for cls in env_config[env]["classes"]]
    # preselected_features = splits_config[env]["filtered_features"]
    all_class_mapping = {key: val for env in env_config["envs"] for key, val in env_config[env]["class_mapping"].items()}
    all_class_to_int_mapping = env_config["class_to_int_mapping"]
    
    feature_names = list(set.union(*[set(splits_config[env]["filtered_features"]) for env in envs]))
    x, y = load_data(scenes, all_classes, feature_names, all_class_mapping, all_class_to_int_mapping)
    test_x, test_y = load_data(test_scenes, all_classes, feature_names, all_class_mapping, all_class_to_int_mapping)
    y, test_y = y.to_numpy(), test_y.to_numpy()


    embedded_x = np.hstack([x[[feature]].to_numpy() for feature in dims_features])
    embedded_test_x = np.hstack([test_x[[feature]].to_numpy() for feature in dims_features])

    print("all eval")
    f1 = evaluate_knn(embedded_x, y, embedded_test_x, test_y, all_class_to_int_mapping)
    

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.concatenate([embedded_x, embedded_test_x])
    y = np.concatenate([y, test_y])
    filtered_tragts = {name: i for name, i in all_class_to_int_mapping.items() if i in y}
    for i, (cls_name, cls_id) in enumerate(filtered_tragts.items()):
        class_mask = y == cls_id
        ax.scatter(
            x[class_mask][:, 0],
            x[class_mask][:, 1],
            x[class_mask][:, 2],
            label=f'{cls_name}',
            s=50
        )
    
    ax.view_init(elev=15, azim=-125) # Set camera
    ax.set_xlabel(f"{env_dims[0]} - {dims_features[0]}")
    ax.set_ylabel(f"{env_dims[1]} - {dims_features[1]}")
    ax.set_zlabel(f"{env_dims[2]} - {dims_features[2]}")
    ax.legend()
    plt.show()