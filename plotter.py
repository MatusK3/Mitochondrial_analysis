import numpy as np
from matplotlib import pyplot as plt

from dataset_loader import DATASETS

from feature_loader import load_features


# cm, corr - Co-occurrence matrix, Correlation
# mrg - aggregation, merged
# ngl - grey level dependencies
# lde - low dependence
# d1 - Offset distance
# fbn - fixed bin number
# n32 - bins

# x_feature = 'cm_auto_corr_d1_3d_v_mrg_fbn_n32' 
# y_feature = 'ngl_lde_d1_a0.0_3d_fbn_n32'


classes = [DATASETS.YPD_SD_Acetate_DAY_3_Acetate, DATASETS.YPD_SD_Acetate_DAY_1_Acetate] 

features, labels = load_features(classes)


# from sklearn.feature_selection import SelectKBest, f_classif
# selector = SelectKBest(score_func=f_classif, k=2)  # choose 2 features
# selector.fit(features, labels)


# mask = selector.get_support()
# selected_features = features.columns[mask]
# print("selected_features:", selected_features)

# un pre processed 
# selected_features = ('morph_asphericity', 'ih_mode_fbn_n32')
# selected_features = ('cm_diff_avg_d1_3d_v_mrg_fbn_n32', 'rlm_lre_3d_v_mrg_fbn_n32')
# selected_features = ('stat_cov', 'cm_info_corr1_d1_3d_v_mrg_fbn_n32')
# selected_features = ('stat_cov', 'rlm_rlnu_norm_3d_v_mrg_fbn_n32')

# pre processed
# selected_features = ('cm_joint_max_d1_3d_v_mrg_fbn_n32', 'dzm_z_perc_3d_fbn_n32')
# selected_features = ('morph_sphericity', 'cm_diff_var_d1_3d_v_mrg_fbn_n32')
selected_features = ('morph_area_dens_conv_hull', 'rlm_sre_3d_v_mrg_fbn_n32')

for i in classes:
    class_mask = labels == i.name
    plt.scatter(features[class_mask][selected_features[0]], features[class_mask][selected_features[1]], label=f'{i.name}')


plt.xlabel(selected_features[0])
plt.ylabel(selected_features[1])
plt.show()


