import numpy as np
from matplotlib import pyplot as plt
import pandas as pd



# cm, corr - Co-occurrence matrix, Correlation
# mrg - aggregation, merged
# ngl - grey level dependencies
# lde - low dependence
# d1 - Offset distance
# fbn - fixed bin number
# n32 - bins

x_feature = 'cm_auto_corr_d1_3d_v_mrg_fbn_n32' 
y_feature = 'ngl_lde_d1_a0.0_3d_fbn_n32'


location = "outputs"
layers = ["SCI_TEST_layer_1.csv", "SCI_TEST_layer_2.csv", "SCI_TEST_layer_3.csv"]
data = [pd.read_csv(location + "/" + layer) for layer in layers]

scene_ids = set().union(*(df['scene_id'].unique() for df in data))
id_colors = {i: np.random.random(3) for i in scene_ids}


fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, layer_data in enumerate(data):
    for scene_id, group in layer_data.groupby('scene_id'):
        axes[i].scatter(
            group[x_feature], group[y_feature],
            color=id_colors[scene_id],
            label=str(scene_id),
            alpha=0.7
        )
    axes[i].set_xlabel(x_feature)
    axes[i].set_ylabel(y_feature)
    axes[i].set_title(f"layer:{i}") 

# fig.suptitle("Scatter plot") # x_feature vs y_feature
plt.tight_layout()
plt.show()

