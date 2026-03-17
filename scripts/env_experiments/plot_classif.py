# snakemake ml_vision conda env
# conda activate .snakemake/conda/7f69f82cf5e2d5fc9f5eb4294544276a_

from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay #confusion_matrix

def show_2d(env, target_names, x, y, features_names):
    filtered_tragts = {name: i for name, i in target_names.items() if i in x or i in y}
    colors = plt.cm.tab10(np.linspace(0, 1, len(filtered_tragts)))

    plt.figure(figsize=(10, 10))
    # fig = plt.figure()
    for i, (cls_name, cls_id) in enumerate(filtered_tragts.items()):
        mask = y == cls_id
        plt.scatter(
            x[mask, 0],
            x[mask, 1],
            color=colors[i],
            label=f'{cls_name}',
            s=60
        )
    plt.title(env)
    plt.xlabel(features_names[0])
    plt.ylabel(features_names[1])
    plt.legend()
    plt.tight_layout()
    plt.show()




def show_3d(target_names, x, y, features_names):
    filtered_tragts = {name: i for name, i in target_names.items() if i in x or i in y}
    colors = plt.cm.tab10(np.linspace(0, 1, len(filtered_tragts)))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, (cls_name, cls_id) in enumerate(filtered_tragts.items()):
        class_mask = y == cls_id

        ax.scatter(
            x[class_mask][:, 0],
            x[class_mask][:, 1],
            x[class_mask][:, 2],
            color=colors[i],
            label=f'{cls_name}',
            s=50
        )
    
    ax.view_init(elev=15, azim=-125) # Set camera
    ax.set_xlabel(features_names[0])
    ax.set_ylabel(features_names[1])
    ax.set_zlabel(features_names[2])
    ax.legend()
    plt.show()

def evaluate_knn(train_x, train_y, test_x, test_y, target_names, features_names, plot=False):
    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    knn.fit(train_x, train_y)
    y_pred = knn.predict(test_x)
    
    print(features_names)
    filtered_tragts = {name: i for name, i in target_names.items() if i in y_pred or i in test_y}
    print(classification_report(test_y, y_pred, target_names=filtered_tragts))

    if plot:
        ConfusionMatrixDisplay.from_predictions(
            test_y,
            y_pred,
            display_labels=target_names.keys(),
            cmap='Blues'
        )

        plt.title(f'Confusion Matrix, knn with features: {features_names}')
        plt.show()


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
    n_features = 2

    top_n_features_file_pth = f"scripts/env_experiments/results/top_{n_features}_mean.csv"
    row = pd.read_csv(f"{top_n_features_file_pth}").iloc[0] # first row of csv
    features_names = row["feature_combination"].split(" ")
    print(features_names, row["mean_f1"])


    
    env_config_pth = "scripts/env_experiments/config.json"
    with open(env_config_pth, 'r') as f:
        env_config = json.load(f)
    envs = env_config["envs"]

    split_config_pth = "scripts/env_experiments/split.json"
    with open(split_config_pth, 'r') as f:
        split_config = json.load(f)

    all_train_x, all_train_y = [], []
    all_val_x, all_val_y = [], []
    all_test_x, all_test_y = [], []
    all_classes = []
    all_class_mappings = {}
    all_class_to_int_mapping = {}
    
    y_shift = 0

    for env in envs:

        print("="*20 + "\n" + env)
        class_to_int_mapping  = env_config[env]["class_to_int_mapping"]


        scenes = split_config[env]["split_scenes"]
        classes = env_config[env]["classes"]
        class_mapping = env_config[env]["class_mapping"]

        train_x, train_y = load_data(scenes["train"], classes, features_names, class_mapping, class_to_int_mapping)
        val_x, val_y = load_data(scenes["val"], classes, features_names, class_mapping, class_to_int_mapping)
        test_x, test_y =load_data(scenes["test"], classes, features_names, class_mapping, class_to_int_mapping)

        # evaluate_knn(train_x, train_y, test_x, test_y, class_to_int_mapping, features_names = features_names)

        # x = np.concatenate((train_x, val_x, test_x), axis=0)
        # y = np.concatenate((train_y, val_y, test_y), axis=0)
        # print(len(x), len(y))
        # if n_features == 2:
        #     show_2d(env, class_to_int_mapping, x, y, features_names)
        # else:
        #     show_3d(class_to_int_mapping, x, y, features_names)

            
        all_train_x.append(train_x)
        all_train_y.append(train_y + y_shift)
        all_val_x.append(val_x)
        all_val_y.append(val_y  + y_shift)
        all_test_x.append(test_x)
        all_test_y.append(test_y  + y_shift)

        all_class_to_int_mapping = {**all_class_to_int_mapping, **{key: val + y_shift for key, val in class_to_int_mapping.items()}}
        all_classes.extend(classes)
        all_class_mappings = {**all_class_mappings, **class_mapping}

        y_shift += len(class_to_int_mapping)
        


    print("="*20 + "\nALL")

    train_x, train_y = np.concatenate(all_train_x), np.concatenate(all_train_y)
    val_x, val_y = np.concatenate(all_val_x), np.concatenate(all_val_y)
    test_x, test_y = np.concatenate(all_test_x), np.concatenate(all_test_y)
    class_to_int_mapping = all_class_to_int_mapping
    classes = all_classes
    class_mappings = all_class_mappings

    print(class_to_int_mapping)
    print(classes)
    print(class_mappings)

    evaluate_knn(train_x, train_y, test_x, test_y, class_to_int_mapping, features_names = features_names)

    x = np.concatenate((train_x, val_x, test_x), axis=0)
    y = np.concatenate((train_y, val_y, test_y), axis=0)
    print(len(x), len(y))
    if n_features == 2:
        show_2d("ALL", class_to_int_mapping, x, y, features_names)
    else:
        show_3d(class_to_int_mapping, x, y, features_names)