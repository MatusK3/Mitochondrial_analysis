from typing import List, Tuple
import pandas as pd
import numpy as np
from config.config import DATASETS, feature_extraction_output_loaction


def load_features(classes : List[DATASETS]) -> Tuple[pd.DataFrame,  pd.DataFrame]:
    data = []
    for i in classes:
        path = f"{feature_extraction_output_loaction}/{i.name}.csv"
        class_data = pd.read_csv(path)
        class_data["class"] = i.name
        data.append(class_data)

    df_data = pd.concat(data, ignore_index=True)

    # filter out metadata, keep only features
    additional_methadata = ["sample_name", "img_name", "mask_index", "class"] # mirp sample_name + my additional columns
    features  = df_data.drop(columns=[c for c in df_data.columns if c.startswith("image_") or c in additional_methadata]) # all metadata from mirp.extract_feature() statrts wit "img"

    features = features.replace([np.inf, -np.inf], np.nan) # get rif of non compatible values
    features = features.dropna(axis=1)  # drop columns with any NaN

    labels = df_data['class']

    return features, labels


def split_pandas_data(features :pd.DataFrame, labels :pd.DataFrame, val_percent :float=0.2, test_percent :float=0.2) -> Tuple[pd.DataFrame,  pd.DataFrame, pd.DataFrame, pd.DataFrame,  pd.DataFrame, pd.DataFrame]:
    data_indices = np.arange(features.shape[0])
    np.random.shuffle(data_indices)

    val_num, test_num = int(features.shape[0] * val_percent), int(features.shape[0] * test_percent)
    train_num = features.shape[0] - val_num - test_num

    train_indices, val_indices, test_indices = data_indices[:train_num], data_indices[train_num:train_num + val_num], data_indices[train_num + val_num : train_num + val_num + test_num]

    x_train, x_val, x_test = features.loc[train_indices], features.loc[val_indices], features.loc[test_indices]
    y_train, y_val, y_test = labels.loc[train_indices], labels.loc[val_indices], labels.loc[test_indices]

    return x_train, y_train, x_val, y_val, x_test, y_test

def split_numpy_data(features :np.ndarray, labels :np.ndarray, val_percent :float=0.2, test_percent :float=0.2) -> Tuple[np.ndarray,  np.ndarray, np.ndarray, np.ndarray,  np.ndarray, np.ndarray]:
    data_indices = np.arange(features.shape[0])
    np.random.shuffle(data_indices)

    val_num, test_num = int(features.shape[0] * val_percent), int(features.shape[0] * test_percent)
    train_num = features.shape[0] - val_num - test_num

    train_indices, val_indices, test_indices = data_indices[:train_num], data_indices[train_num:train_num + val_num], data_indices[train_num + val_num : train_num + val_num + test_num]

    x_train, x_val, x_test = features[train_indices], features[val_indices], features[test_indices]
    y_train, y_val, y_test = labels[train_indices], labels[val_indices], labels[test_indices]

    return x_train, y_train, x_val, y_val, x_test, y_test




if __name__ =="__main__":
    ...


    




