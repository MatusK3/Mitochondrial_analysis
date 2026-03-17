import argparse
from itertools import combinations

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from utils.feature_loader import load_features, DATASETS_SPLITS

def select_features(num_of_features_to_select: int, pre_selected_features_pth: str, data_dir: str):
    
    with open(pre_selected_features_pth, 'r') as file:
        line = file.readline().strip()
    preslected_f = line.split(" ")

    data = load_features(data_dir, [DATASETS_SPLITS.TRAIN, DATASETS_SPLITS.VAL], preslected_f)
    train_x, train_y = data[DATASETS_SPLITS.TRAIN]["x"], data[DATASETS_SPLITS.TRAIN]["y"]
    val_x, val_y = data[DATASETS_SPLITS.VAL]["x"], data[DATASETS_SPLITS.VAL]["y"]
    train_y, val_y = train_y.to_numpy(), val_y.to_numpy()

    # preslected_f = train_x.columns.values # load all names, in case if preslected_f was orignaly None
    if num_of_features_to_select == 0: # select all
        num_of_features_to_select = len(preslected_f)
    

    data_collection = []

    column_combinations = list(combinations(preslected_f, num_of_features_to_select))
    column_combinations_pbar = tqdm(column_combinations, desc=f"iterating feature combination: {num_of_features_to_select} out of {len(preslected_f)}", unit="comb")
    for feature_combination in column_combinations_pbar:
        feature_combination = list(feature_combination)

        feature_subse_train_x = train_x[feature_combination].to_numpy()
        feature_subse_val_x = val_x[feature_combination].to_numpy()

        X = np.concatenate([feature_subse_train_x, feature_subse_val_x])
        y = np.concatenate([train_y, val_y])

        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=8, random_state=42)

        knn = KNeighborsClassifier(n_neighbors=5, weights='uniform') # uniform should help a little for imbalanced dataset
        # f1_macro; f1_weighted; f1_micro
        scores = cross_val_score(knn, X, y, cv=rskf, scoring='f1_macro', n_jobs=5) # f1_macro for imbalanced datasets

        mean_f1 = scores.mean()
        std_f1 = scores.std()
        # print(f"Mean f1_macro: {mean_f1:.4f} (std f1_macro: {std_f1})")

        data_collection.append({
            "feature_combination": " ".join(feature_combination),
            "mean_f1_macro": mean_f1,
            "std_f1_macro": std_f1
        })

    return data_collection


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dark/light image pairs.")
    
    parser.add_argument("--select_n_features", type=int, required=True, help="num, how many features to select")
    parser.add_argument("--pre_selected_features", type=str, required=True, help="path to txt file, with pre selected features")
    parser.add_argument("--data_dir", type=str, required=True, help="path to dir, with csv of all fearures")
    parser.add_argument("--output", type=str, required=True, help="path to csv file, for output")

    args = parser.parse_args()

    data_collection = select_features(args.select_n_features, args.pre_selected_features, args.data_dir)

    df = pd.DataFrame(data_collection)
    TOP_N = 50
    df_sorted = df.sort_values(by=['mean_f1_macro', 'std_f1_macro'], ascending=[False, True])
    top = df_sorted.head(TOP_N)
    top.to_csv(args.output, index=False)
