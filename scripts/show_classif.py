from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from utils.feature_loader import load_features, DATASETS_SPLITS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay #confusion_matrix

def show_2d(target_names, x, y, features_names):
    ...

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
config_path = Path.joinpath(DATASET_PATH, "config.json")
if __name__ == "__main__":
    with open(config_path, 'r') as f:
        config = json.load(f)
    target_names  = config["class_to_int_mapping"]

    n_features = 3#2



    top_n_features_file_pth = f"results/features/top/selected_{n_features}.csv"
    features_names, val_f1, val_f1_std = pd.read_csv(f"{top_n_features_file_pth}").iloc[0] # first row of csv

    
    # OLD BUGGED FEATURES
    # features_names = "cm_mcc_d1_2d_avg_fbn_n32 szm_glnu_norm_2d_fbn_n32 lbp_2d_rot_invar_d2_cm_info_corr1_d1_2d_s_mrg_fbn_n32" # all
    # features_names = "lbp_2d_rot_invar_d2_cm_info_corr1_d1_2d_s_mrg_fbn_n32 cm_corr_d1_2d_avg_fbn_n32 cm_inv_diff_norm_d1_2d_avg_fbn_n32" # all, downsample
    # features_names = "cm_mcc_d1_2d_avg_fbn_n32 ngl_lgce_d1_a0.0_2d_fbn_n32 lbp_2d_rot_invar_d2_cm_info_corr1_d1_2d_s_mrg_fbn_n32" # no day2
    # features_names = "lbp_2d_rot_invar_d2_cm_info_corr2_d1_2d_s_mrg_fbn_n32 morph_area_dens_aabb szm_glnu_norm_2d_fbn_n32" # only new classes: ["YPD%1_DAY1", "YPD%1_DAY2", "YPD%1_DAY3", "YPGly_DAY1", "YPGly_DAY2", "YPGly_DAY3", "YPGal_DAY1", "YPGal_DAY2", "YPGal_DAY3"]
    # features_names = "szm_zs_entr_2d_fbn_n32 ngl_dc_var_d1_a0.0_2d_fbn_n32 ih_medad_fbn_n32" # old only
    # features_names = "lbp_2d_rot_invar_d2_cm_info_corr2_d1_2d_s_mrg_fbn_n32 szm_glnu_norm_2d_fbn_n32 cm_energy_d1_2d_avg_fbn_n32" # only new, no day2: ["YPD%1_DAY1", "YPD%1_DAY3", "YPGly_DAY1", "YPGly_DAY3", "YPGal_DAY1", "YPGal_DAY3"]
    # features_names = features_names.split(" ")


    # NEW
    # features_names = "cm_mcc_d1_2d_avg_fbn_n32 lbp_2d_rot_invar_d2_dzm_sde_2d_fbn_n32 lbp_2d_rot_invar_d2_ivh_v25" # all



    features_names = features_names.split(" ")
    print(features_names, val_f1, val_f1_std)

    data_dir = "results/features/mirp"
    data = load_features(data_dir, [DATASETS_SPLITS.TRAIN, DATASETS_SPLITS.VAL, DATASETS_SPLITS.TEST], features_names)
    train_x, train_y = data[DATASETS_SPLITS.TRAIN]["x"].to_numpy(), data[DATASETS_SPLITS.TRAIN]["y"].to_numpy()
    val_x, val_y = data[DATASETS_SPLITS.VAL]["x"].to_numpy(), data[DATASETS_SPLITS.VAL]["y"].to_numpy()
    test_x, test_y = data[DATASETS_SPLITS.TEST]["x"].to_numpy(), data[DATASETS_SPLITS.TEST]["y"].to_numpy()

    evaluate_knn(train_x, train_y, test_x, test_y, target_names, features_names = features_names)


    x = np.concatenate((train_x, val_x, test_x), axis=0)
    y = np.concatenate((train_y, val_y, test_y), axis=0)
    print(len(x), len(y))
    if n_features == 2:
        show_2d(target_names, x, y, features_names)
    else:
        show_3d(target_names, x, y, features_names)