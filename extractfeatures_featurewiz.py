import numpy as np

from config.config import DATASETS
from feature_loader import load_features, split_pandas_data
from classification import knn

from featurewiz import FeatureWiz


if __name__ =="__main__":
    classes = [DATASETS.YPD_SD_Acetate_DAY_3_Acetate, DATASETS.YPD_SD_Acetate_DAY_1_Acetate] 


    dataset, labels = load_features(classes)
    x_train, y_train, x_val, y_val, x_test, y_test = split_pandas_data(dataset, labels, val_percent=0.3, test_percent=0)


    f_wiz = FeatureWiz(
        feature_engg='',
        corr_limit=0.9,  
        nrows=None,
        transform_target=False,
        category_encoders="auto",
        verbose=0 #2
    )

    x_train_selected, y_train_selected = f_wiz.fit_transform(x_train, y_train)

    selected_features = f_wiz.features
    print(len(selected_features))
    print(selected_features)




    # calculate acc
    x_val_selected, y_val_selected = f_wiz.transform(x_val), y_val.copy()

    x_train_selected, y_train_selected = x_train_selected.to_numpy(), y_train_selected.to_numpy()
    x_val_selected, y_val_selected = x_val_selected.to_numpy(), y_val_selected.to_numpy()


    for k in [3, 5, 7, 9, 11, 13]:
        acc = 0
        correct = 0
        for input, target in zip(x_val_selected, y_val_selected):
            prediction = knn(input, k, x_train_selected, y_train_selected)
            if prediction == target:
                correct += 1
        acc = correct / len(x_val)
        print(f"k: {k}, acc:{acc}")

# k: 3, acc:0.6206896551724138
# k: 5, acc:0.6551724137931034
# k: 7, acc:0.6436781609195402
# k: 9, acc:0.7011494252873564
# k: 11, acc:0.6896551724137931
# k: 13, acc:0.6551724137931034
