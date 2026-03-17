

from sklearn.manifold import TSNE

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from utils.feature_loader import load_features, DATASETS_SPLITS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay #confusion_matrix

def evaluate_knn(train_x, train_y, test_x, test_y, target_names, features_names, plot=False):
    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    knn.fit(train_x, train_y)
    y_pred = knn.predict(test_x)
    
    # print(features_names)
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


if __name__ == "__main__":
    DATASET_PATH = Path("data/mito")

    config_path = Path.joinpath(DATASET_PATH, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    class_to_int_mapping = config["class_to_int_mapping"]
    int_to_class = {v: k for k, v in class_to_int_mapping.items()}

    data_dir = "results/features/mirp"

    all_data_featurewiz_selection = "ih_skew_fbn_n32 cm_clust_prom_d2_2d_avg_fbn_n32 cm_corr_d2_2d_avg_fbn_n32 lbp_2d_rot_invar_d2_rlm_srlge_2d_s_mrg_fbn_n32 lbp_2d_rot_invar_d2_cm_inv_var_d1_2d_s_mrg_fbn_n32 ih_medad_fbn_n32 lbp_2d_rot_invar_d2_dzm_hgze_2d_fbn_n32 szm_lzlge_2d_fbn_n32 cm_mcc_d1_2d_avg_fbn_n32 cm_diff_var_d2_2d_avg_fbn_n32 lbp_2d_rot_invar_d2_rlm_lgre_2d_s_mrg_fbn_n32 morph_pca_elongation szm_zsnu_norm_2d_fbn_n32 lbp_2d_rot_invar_d2_ih_rmad_fbn_n32 cm_diff_entr_d2_2d_avg_fbn_n32 ivh_diff_v25_v75 morph_pca_min_axis morph_vol_dens_aabb morph_vol_dens_mvee lbp_2d_rot_invar_d2_ngl_dc_energy_d1_a0.0_2d_fbn_n32 lbp_2d_rot_invar_d2_rlm_lrhge_2d_s_mrg_fbn_n32 ngl_hdhge_d1_a0.0_2d_fbn_n32 lbp_2d_rot_invar_d2_dzm_sde_2d_fbn_n32 lbp_2d_rot_invar_d2_ngl_lgce_d1_a0.0_2d_fbn_n32 lbp_2d_rot_invar_d2_cm_mcc_d1_2d_s_mrg_fbn_n32 lbp_2d_rot_invar_d2_szm_glnu_norm_2d_fbn_n32 ngl_dc_entr_d1_a0.0_2d_fbn_n32 lbp_2d_rot_invar_d2_stat_cov morph_area_dens_conv_hull cm_info_corr1_d2_2d_avg_fbn_n32 lbp_2d_rot_invar_d2_rlm_rl_var_2d_s_mrg_fbn_n32 lbp_2d_rot_invar_d2_ngl_hdhge_d1_a0.0_2d_fbn_n32 dzm_sdhge_2d_fbn_n32 morph_vol_dens_ombb lbp_2d_rot_invar_d2_rlm_glnu_norm_2d_s_mrg_fbn_n32 ih_cov_fbn_n32 lbp_2d_rot_invar_d2_ivh_v25 lbp_2d_rot_invar_d2_szm_zs_var_2d_fbn_n32 lbp_2d_rot_invar_d2_ngl_dc_var_d1_a0.0_2d_fbn_n32 dzm_zdnu_norm_2d_fbn_n32 dzm_gl_var_2d_fbn_n32 ngt_strength_2d_fbn_n32 lbp_2d_rot_invar_d2_szm_zsnu_norm_2d_fbn_n32 lbp_2d_rot_invar_d2_dzm_lgze_2d_fbn_n32 szm_hgze_2d_fbn_n32 lbp_2d_rot_invar_d2_ivh_diff_v10_v90 rlm_srlge_2d_avg_fbn_n32 lbp_2d_rot_invar_d2_ivh_i90 ngl_dc_var_d2_a0.0_2d_fbn_n32 ngl_dcnu_norm_d1_a0.0_2d_fbn_n32 lbp_2d_rot_invar_d2_ngl_ldhge_d1_a0.0_2d_fbn_n32 lbp_2d_rot_invar_d2_stat_rms cm_joint_max_d1_2d_avg_fbn_n32 lbp_2d_rot_invar_d2_cm_diff_var_d1_2d_s_mrg_fbn_n32 cm_energy_d2_2d_avg_fbn_n32 ngl_ldlge_d2_a0.0_2d_fbn_n32 dzm_glnu_norm_2d_fbn_n32 lbp_2d_rot_invar_d2_dzm_zdnu_2d_fbn_n32 szm_szlge_2d_fbn_n32 lbp_2d_rot_invar_d2_rlm_lrlge_2d_s_mrg_fbn_n32 lbp_2d_rot_invar_d2_szm_z_perc_2d_fbn_n32 lbp_2d_rot_invar_d2_ivh_v50 rlm_rl_var_2d_avg_fbn_n32 cm_inv_diff_d2_2d_avg_fbn_n32 lbp_2d_rot_invar_d2_cm_sum_entr_d1_2d_s_mrg_fbn_n32 ngl_ldhge_d1_a0.0_2d_fbn_n32 cm_joint_entr_d2_2d_avg_fbn_n32 lbp_2d_rot_invar_d2_cm_clust_prom_d1_2d_s_mrg_fbn_n32 lbp_2d_rot_invar_d2_ngl_dcnu_norm_d1_a0.0_2d_fbn_n32 lbp_2d_rot_invar_d2_loc_peak_glob lbp_2d_rot_invar_d2_ivh_i50"
    all_data_featurewiz_selection = all_data_featurewiz_selection.split(" ")

    data = load_features(data_dir, [DATASETS_SPLITS.TRAIN, DATASETS_SPLITS.VAL, DATASETS_SPLITS.TEST], all_data_featurewiz_selection)
    train_x, train_y = data[DATASETS_SPLITS.TRAIN]["x"].to_numpy(), data[DATASETS_SPLITS.TRAIN]["y"].to_numpy()
    val_x, val_y = data[DATASETS_SPLITS.VAL]["x"].to_numpy(), data[DATASETS_SPLITS.VAL]["y"].to_numpy()
    test_x, test_y = data[DATASETS_SPLITS.TEST]["x"].to_numpy(), data[DATASETS_SPLITS.TEST]["y"].to_numpy()


    x = np.concatenate([train_x, val_x])
    y = np.concatenate([train_y, val_y])


    for i in range(1, 7):
        print(f"n_components: {i}")
        # dim = 6
        dim = i
        
        lda = LinearDiscriminantAnalysis(n_components=dim)
        X_embedded = lda.fit_transform(x, y)
        # if dim == 2:
        #     plt.figure(figsize=(10, 10))
        #     for class_int in np.unique(y):
        #         mask = y == class_int
        #         plt.scatter(
        #             X_embedded[mask, 0],
        #             X_embedded[mask, 1],
        #             label=int_to_class[class_int],
        #             s=60
        #         )
        #     plt.title("LDA 2D Projection")
        #     plt.xlabel("LDA-1")
        #     plt.ylabel("LDA-2")
        #     plt.legend()

        # if dim == 3:        
        #     fig = plt.figure(figsize=(10, 10))
        #     ax = fig.add_subplot(111, projection='3d')
        #     for class_int in np.unique(y):
        #         mask = y == class_int
        #         ax.scatter(
        #             X_embedded[mask, 0],
        #             X_embedded[mask, 1],
        #             X_embedded[mask, 2],
        #             label=int_to_class[class_int],
        #             s=60
        #         )

        #     ax.set_title("LDA 3D Projection")
        #     ax.set_xlabel("LDA-1")
        #     ax.set_ylabel("LDA-2")
        #     ax.set_zlabel("LDA-3")

        #     ax.legend()

        test_x_embedded = lda.transform(test_x)
        evaluate_knn(X_embedded, y, test_x_embedded, test_y, target_names=class_to_int_mapping, features_names = [f"LDA-{i}" for i in range(dim)])


        # if dim == 2 or dim == 3:
        #     plt.tight_layout()
        #     plt.show()


                