import argparse

from featurewiz import FeatureWiz

from utils.feature_loader import load_features, DATASETS_SPLITS

# DATASET_PATH = Path("data/mito")
# config_path = Path.joinpath(DATASET_PATH, "config.json")


def get_filtered_features(data_dir):
    # with open(config_path, 'r') as f:
    #     config = json.load(f)
    # dataset_structure = pd.read_csv(Path.joinpath(DATASET_PATH, config["scans_list_path"]))

    data = load_features(data_dir, [DATASETS_SPLITS.TRAIN])
    print(data)
    x_train, y_train = data[DATASETS_SPLITS.TRAIN]["x"], data[DATASETS_SPLITS.TRAIN]["y"]

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

    print(f"num of features before: {x_train.shape[1]}; after: {len(selected_features)}")

    return selected_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dark/light image pairs.")

    # parser.add_argument("--train_data_pth", required=True, help="path to csv file, with training features")
    parser.add_argument("--extr_ftr_dir", required=True, help="path to folder, with extracted featres")
    parser.add_argument("--output", required=True, help="path to output txt file")

    args = parser.parse_args()

    filtered_features = get_filtered_features(args.extr_ftr_dir)

    with open(args.output, "w") as f:
        f.write(" ".join(filtered_features))