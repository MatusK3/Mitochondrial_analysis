from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


from config.config import DATASETS, feature_extraction_output_loaction


def load_features(classes : List[DATASETS]) -> Tuple[pd.DataFrame,  pd.DataFrame]:
    data = []
    for i in classes:
        # path = f"{feature_extraction_output_loaction}/un_preprocessed/{i.name}.csv"  # unprocessed data 
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


# def split_pandas_data(features :pd.DataFrame, labels :pd.DataFrame, val_percent :float=0.2, test_percent :float=0.2) -> Tuple[pd.DataFrame,  pd.DataFrame, pd.DataFrame, pd.DataFrame,  pd.DataFrame, pd.DataFrame]:
#     data_indices = np.arange(features.shape[0])
#     np.random.shuffle(data_indices)

#     val_num, test_num = int(features.shape[0] * val_percent), int(features.shape[0] * test_percent)
#     train_num = features.shape[0] - val_num - test_num

#     train_indices, val_indices, test_indices = data_indices[:train_num], data_indices[train_num:train_num + val_num], data_indices[train_num + val_num : train_num + val_num + test_num]

#     x_train, x_val, x_test = features.loc[train_indices], features.loc[val_indices], features.loc[test_indices]
#     y_train, y_val, y_test = labels.loc[train_indices], labels.loc[val_indices], labels.loc[test_indices]

#     return x_train, y_train, x_val, y_val, x_test, y_test

def split_pandas_data(features, labels, val_percent=0.2, test_percent=0.2):
    # First split off test
    if test_percent > 0:
        x_train, x_test, y_train, y_test = train_test_split(
            features, labels,
            test_size=test_percent,
            stratify=labels if labels.nunique().min() > 1 else None,
            shuffle=True
        )
    else:
        # No test set
        x_train, y_train = features, labels
        x_test = features.iloc[0:0]   # empty DataFrame with correct columns
        y_test = labels.iloc[0:0]

    # Now split train into train/val
    val_ratio_adjusted = val_percent / (1 - test_percent)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train,
        test_size=val_ratio_adjusted,
        stratify=y_train,
        shuffle=True
    )

    return x_train, y_train, x_val, y_val, x_test, y_test

# def split_numpy_data(features :np.ndarray, labels :np.ndarray, val_percent :float=0.2, test_percent :float=0.2) -> Tuple[np.ndarray,  np.ndarray, np.ndarray, np.ndarray,  np.ndarray, np.ndarray]:
#     data_indices = np.arange(features.shape[0])
#     np.random.shuffle(data_indices)

#     val_num, test_num = int(features.shape[0] * val_percent), int(features.shape[0] * test_percent)
#     train_num = features.shape[0] - val_num - test_num

#     train_indices, val_indices, test_indices = data_indices[:train_num], data_indices[train_num:train_num + val_num], data_indices[train_num + val_num : train_num + val_num + test_num]

#     x_train, x_val, x_test = features[train_indices], features[val_indices], features[test_indices]
#     y_train, y_val, y_test = labels[train_indices], labels[val_indices], labels[test_indices]

#     return x_train, y_train, x_val, y_val, x_test, y_test


def split_numpy_data(features: np.ndarray, labels: np.ndarray, val_percent: float = 0.2, test_percent: float = 0.2) -> Tuple[np.ndarray, np.ndarray,
                                                                                                                            np.ndarray, np.ndarray,
                                                                                                                            np.ndarray, np.ndarray]:
    # 1) First split off test set
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_percent,
        stratify=labels,
        shuffle=True,
    )
    # 2) Compute adjusted validation split relative to the remaining data
    val_ratio_adjusted = val_percent / (1 - test_percent)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=val_ratio_adjusted,
        stratify=y_train,
        shuffle=True,
    )

    return x_train, y_train, x_val, y_val, x_test, y_test




def stratified_kfold_split(features: np.ndarray,
                           labels: np.ndarray,
                           n_splits: int = 10,
                           seed: int = 42):

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
        x_train, x_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        yield fold, x_train, y_train, x_val, y_val

if __name__ =="__main__":
    ...


    




