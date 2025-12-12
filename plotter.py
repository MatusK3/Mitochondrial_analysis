import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from config.config import DATASETS
from feature_loader import load_features



# classes = [DATASETS.YPD_SD_Acetate_DAY_3_Acetate, DATASETS.YPD_SD_Acetate_DAY_1_Acetate] 
classes = [
        DATASETS.YPD_SD_Acetate_DAY_1_Acetate, DATASETS.YPD_SD_Acetate_DAY_3_Acetate,
        DATASETS.YPD_SD_Acetate_DAY_1_SD, DATASETS.YPD_SD_Acetate_DAY_3_SD,
        DATASETS.YPD_SD_Acetate_DAY_1_YPD, DATASETS.YPD_SD_Acetate_DAY_3_YPD
    ] 

features, labels = load_features(classes)


# SEARCH
# #selected_dataset = "brute_force_2_features.csv"
# selected_dataset = "brute_force_3_features.csv"
# selected_dataset = "bf_Acetate_SD_YPD_D1_D3_2_features.csv"
# df = pd.read_csv(selected_dataset)
# df= df.sort_values(by='acc', ascending=False)

# df_filtered = df[~(
#     df['feature_0'].str.contains('st', case=False, na=False) |
#     df['feature_1'].str.contains('st', case=False, na=False)
# )]


# for id in range(10):
#     first_row = df.iloc[id]
#     # Access individual values
#     selected_features = tuple(first_row.drop('acc'))
#     val_acc = first_row['acc']
#     print(selected_features, val_acc)
#     break

# selected_features = ('stat_min', 'cm_diff_var_d1_2d_avg_fbn_n32', 'lbp_2d_rot_invar_d1_dzm_zdnu_norm_2d_fbn_n32')
# val_acc = 0.896551724137931
selected_features = ('lbp_2d_rot_invar_d1_ngl_lgce_d1_a0.0_2d_fbn_n32', 'cm_corr_d1_2d_avg_fbn_n32', 'morph_area_dens_conv_hull')
val_acc = 0.6796540880503145

# selected_features = ('lbp_2d_rot_invar_d1_ngl_lgce_d1_a0.0_2d_fbn_n32', 'cm_corr_d1_2d_avg_fbn_n32')
# val_acc = 0.5782075471698114

# human_readable_feature_names = ('normalized instensity, min.', 'co-occurrence variance', 'LBP, norm. zone distance')

if len(selected_features) == 2:
    for i, cls in enumerate(classes):
        class_mask = labels == cls.name
        plt.scatter(features[class_mask][selected_features[0]], features[class_mask][selected_features[1]], label=f'{cls.name}')
    plt.xlabel(selected_features[0])
    plt.ylabel(selected_features[1])
    plt.title(f"classification accuracy {val_acc:.3f}")
    plt.legend()
    plt.show()

if len(selected_features) == 3:
    from mpl_toolkits.mplot3d import Axes3D
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, cls in enumerate(classes):
        class_mask = labels == cls.name
        ax.scatter(features[class_mask][selected_features[0]], 
                features[class_mask][selected_features[1]], 
                features[class_mask][selected_features[2]], label=f'{cls.name}')
        ax.view_init(elev=15, azim=-125)
    ax.set_xlabel(selected_features[0])
    ax.set_ylabel(selected_features[1])
    ax.set_zlabel(selected_features[2])
    # ax.set_xlabel(human_readable_feature_names[0])
    # ax.set_ylabel(human_readable_feature_names[1])
    # ax.set_zlabel(human_readable_feature_names[2])

    # Turn off tick labels
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_zticklabels([])
    # or other variant
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])


    ax.set_title(f"classification accuracy {val_acc:.3f}")


    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # ax.xaxis.pane.set_edgecolor('w')
    # ax.yaxis.pane.set_edgecolor('w')
    # ax.zaxis.pane.set_edgecolor('w')

    # ax.xaxis.pane.fill = False
    # ax.yaxis.pane.fill = False
    # ax.zaxis.pane.fill = False

    plt.tight_layout()
    plt.legend()

    plt.savefig('output.pdf', bbox_inches='tight')

    plt.show()


        


