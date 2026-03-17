# snakemake ml_vision conda env
# conda activate .snakemake/conda/7f69f82cf5e2d5fc9f5eb4294544276a_

import json
from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

    filtered_tragts = {name: i for name, i in target_names.items() if i in y_pred or i in test_y}
    print(classification_report(test_y, y_pred, target_names=filtered_tragts))
    return f1_score(test_y, y_pred, average="macro")



if __name__ == "__main__":

    env_config_pth = "scripts/env_experiments/config.json"
    with open(env_config_pth, 'r') as f:    
        env_config = json.load(f)
    envs = env_config["envs"]

    splits_config_pth = "scripts/env_experiments/split.json"
    with open(splits_config_pth, 'r') as f:
        splits_config = json.load(f)


    for env in envs:
        scenes = np.concatenate([
            splits_config[env]["split_scenes"]["train"],
            splits_config[env]["split_scenes"]["val"]
        ])
        test_scenes = splits_config[env]["split_scenes"]["test"]

        
        
        classes = env_config[env]["classes"]
        preselected_features = splits_config[env]["filtered_features"]
        class_mapping = env_config[env]["class_mapping"]
        class_to_int_mapping = env_config[env]["class_to_int_mapping"]
        
        int_to_class_mapping = {v: k for k, v in class_to_int_mapping.items()}


        x, y = load_data(scenes, classes, preselected_features, class_mapping, class_to_int_mapping)
        x, y = x.to_numpy(), y.to_numpy()
        
        test_x, test_y = load_data(test_scenes, classes, preselected_features, class_mapping, class_to_int_mapping)
        test_x, test_y = test_x.to_numpy(), test_y.to_numpy()

        # tsne = TSNE(
        #     n_components = 2,
        #     perplexity = 100, #30,
        #     learning_rate = 'auto',
        #     max_iter = 2000, #1000,
        #     n_iter_without_progress = 500, #300
        #     init = 'pca',
        #     random_state = 42,
        #     method = 'barnes_hut', #'exact',
        #     n_jobs = 5
        # )
        # X_embedded = tsne.fit_transform(x)
        
        # plt.figure(figsize=(10, 10))
        # for class_int in np.unique(y):
        #     mask = y == class_int
        #     plt.scatter(
        #         X_embedded[mask, 0],
        #         X_embedded[mask, 1],
        #         label=int_to_class_mapping[class_int],
        #         s=60
        #     )
        # plt.title(f"{env} t-SNE 2D Projection")
        # plt.xlabel("TSNE-1")
        # plt.ylabel("TSNE-2")
        # plt.legend()
        # plt.tight_layout()
        # plt.show()


        dim = 1
        if dim == 1:
            lda = LinearDiscriminantAnalysis(n_components=dim)
            X_embedded = lda.fit_transform(x, y)

            test_x_embeded = lda.transform(test_x)
            f1 = evaluate_knn(X_embedded, y, test_x_embeded, test_y, class_to_int_mapping)

            plt.figure(figsize=(8, 4))
            for class_int in np.unique(y):
                mask = y == class_int
                plt.hist(X_embedded[mask], label=int_to_class_mapping[class_int], alpha=0.5, bins=30)

            plt.title(f"{env} LDA; test: {f1}")
            plt.xlabel("LDA Component 1")
            plt.legend()
            plt.show()

        if dim == 2:
            # tsne = TSNE(
            #     n_components = 2,
            #     perplexity = 50, #30,
            #     learning_rate = 'auto',
            #     max_iter = 2000, #1000,
            #     n_iter_without_progress = 500, #300
            #     init = 'pca',
            #     random_state = 42,
            #     method = 'barnes_hut', #'exact',
            #     n_jobs = 5
            # )
            # X_embedded = tsne.fit_transform(x)

            lda = LinearDiscriminantAnalysis(n_components=2)
            X_embedded = lda.fit_transform(x, y)
            
            plt.figure(figsize=(12, 12))
            for class_int in np.unique(y):
                mask = y == class_int
                plt.scatter(
                    X_embedded[mask, 0],
                    X_embedded[mask, 1],
                    label=int_to_class_mapping[class_int],
                    s=60
                )
            plt.title(f"{env} t-SNE 2D Projection")
            plt.xlabel("TSNE-1")
            plt.ylabel("TSNE-2")
            plt.legend()
            plt.tight_layout()
            plt.show()

        # if dim == 3:
        #     tsne = TSNE(
        #         n_components = 3,
        #         perplexity = 50, #30,
        #         learning_rate = 'auto',
        #         max_iter = 2000, #1000,
        #         n_iter_without_progress = 500, #300
        #         init = 'pca',
        #         random_state = 42,
        #         method = 'barnes_hut', #'exact',
        #         n_jobs = 5
        #     )
        #     X_embedded = tsne.fit_transform(x)
            
        #     fig = plt.figure(figsize=(15, 15))
        #     ax = fig.add_subplot(111, projection='3d')
        #     for class_int in np.unique(y):
        #         mask = y == class_int
        #         ax.scatter(
        #             X_embedded[mask, 0],
        #             X_embedded[mask, 1],
        #             X_embedded[mask, 2],
        #             label=int_to_class_mapping[class_int],
        #             s=60
        #         )

        #     ax.set_title(f"{env} 3D t-SNE Projection")
        #     ax.set_xlabel("TSNE-1")
        #     ax.set_ylabel("TSNE-2")
        #     ax.set_zlabel("TSNE-3")

        #     ax.legend()

        #     plt.tight_layout()
        #     plt.show()