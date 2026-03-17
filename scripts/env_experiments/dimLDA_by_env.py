# snakemake ml_vision conda env
# conda activate .snakemake/conda/7f69f82cf5e2d5fc9f5eb4294544276a_

import json
from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay #confusion_matrix
from sklearn.metrics import f1_score




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



if __name__ == "__main__":
    env_config_pth = "scripts/env_experiments/config.json"
    # env_config_pth = "scripts/env_experiments/config_ideal.json"
    with open(env_config_pth, 'r') as f:    
        env_config = json.load(f)
    envs = env_config["envs"]

    splits_config_pth = "scripts/env_experiments/split.json"
    with open(splits_config_pth, 'r') as f:
        splits_config = json.load(f)

    feature_names = list(set.union(*[set(splits_config[env]["filtered_features"]) for env in envs]))

    env_dims = ["YPD", "YPGly", "YPGal"]
    dims_transforms = []
    for env in env_dims:
        scenes = np.concatenate([
            splits_config[env]["split_scenes"]["train"],
            splits_config[env]["split_scenes"]["val"]
        ])
        test_scenes = splits_config[env]["split_scenes"]["test"]

        
        
        classes = env_config[env]["classes"]
        # preselected_features = splits_config[env]["filtered_features"]
        class_mapping = env_config[env]["class_mapping"]
        class_to_int_mapping = env_config[env]["class_to_int_mapping"]
        
        int_to_class_mapping = {v: k for k, v in class_to_int_mapping.items()}


        x, y = load_data(scenes, classes, feature_names, class_mapping, class_to_int_mapping)
        x, y = x.to_numpy(), y.to_numpy()
        test_x, test_y = load_data(test_scenes, classes, feature_names, class_mapping, class_to_int_mapping)
        test_x, test_y = test_x.to_numpy(), test_y.to_numpy()

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        test_x_scaled = scaler.transform(test_x)


        lda = LinearDiscriminantAnalysis(n_components=1)
        X_embedded = lda.fit_transform(x_scaled, y)
        dims_transforms.append(lda.transform)

        test_x_embeded = lda.transform(test_x_scaled)
        f1 = evaluate_knn(X_embedded, y, test_x_embeded, test_y, class_to_int_mapping)

        plt.figure(figsize=(8, 8))
        for class_int in np.unique(y):
            mask = y == class_int
            plt.hist(X_embedded[mask], label=int_to_class_mapping[class_int], alpha=0.5, bins=30)

        plt.title(f"{env} LDA; test: {f1}")
        plt.xlabel("LDA Component 1")
        plt.legend()
        print(env)
        

    
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
    

    x, y = load_data(scenes, all_classes, feature_names, all_class_mapping, all_class_to_int_mapping)
    x, y = x.to_numpy(), y.to_numpy()
    test_x, test_y = load_data(test_scenes, all_classes, feature_names, all_class_mapping, all_class_to_int_mapping)
    test_x, test_y = test_x.to_numpy(), test_y.to_numpy()

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    test_x_scaled = scaler.transform(test_x)

    embedded_x = np.hstack([transform(x_scaled) for transform in dims_transforms])
    embedded_test_x = np.hstack([transform(test_x_scaled) for transform in dims_transforms])

    print("all eval")
    f1 = evaluate_knn(embedded_x, y, embedded_test_x, test_y, all_class_to_int_mapping)
    
    # plot
    filtered_tragts = {name: i for name, i in all_class_to_int_mapping.items() if i in y}
    # colors = plt.cm.tab10(np.linspace(0, 1, len(filtered_tragts)))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.concatenate([embedded_x, embedded_test_x])
    y = np.concatenate([y, test_y])
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
    ax.set_xlabel(env_dims[0])
    ax.set_ylabel(env_dims[1])
    ax.set_zlabel(env_dims[2])
    ax.legend()
    # plt.show()
        


    # all - ALL
    # scenes = np.concatenate([
    #         np.concatenate([
    #             splits_config[env]["split_scenes"]["train"],
    #             splits_config[env]["split_scenes"]["val"]
    #         ])for env in env_dims
    #     ])
    # test_scenes = np.concatenate([
    #     splits_config[env]["split_scenes"]["test"] for env in env_dims
    # ])

    
    
    # all_classes = [cls for env in env_dims for cls in env_config[env]["classes"]]
    # # preselected_features = splits_config[env]["filtered_features"]
    # all_class_mapping = {key: val for env in env_dims for key, val in env_config[env]["class_mapping"].items()}
    # all_class_to_int_mapping = env_config["class_to_int_mapping"]
    

    # x, y = load_data(scenes, all_classes, feature_names, all_class_mapping, all_class_to_int_mapping)
    # x, y = x.to_numpy(), y.to_numpy()
    # test_x, test_y = load_data(test_scenes, all_classes, feature_names, all_class_mapping, all_class_to_int_mapping)
    # test_x, test_y = test_x.to_numpy(), test_y.to_numpy()

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


    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    test_x_scaled = scaler.transform(test_x)

    embedded_x = np.hstack([transform(x_scaled) for transform in dims_transforms])
    embedded_test_x = np.hstack([transform(test_x_scaled) for transform in dims_transforms])

    print("all eval")
    f1 = evaluate_knn(embedded_x, y, embedded_test_x, test_y, all_class_to_int_mapping)
    
    # plot
    filtered_tragts = {name: i for name, i in all_class_to_int_mapping.items() if i in y}
    # colors = plt.cm.tab10(np.linspace(0, 1, len(filtered_tragts)))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.concatenate([embedded_x, embedded_test_x])
    y = np.concatenate([y, test_y])
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
    ax.set_xlabel(env_dims[0])
    ax.set_ylabel(env_dims[1])
    ax.set_zlabel(env_dims[2])
    ax.legend()
    plt.show()