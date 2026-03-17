import pandas as pd
from pathlib import Path
import json

DATASET_PATH = Path("data/mito")

config_path = Path.joinpath(DATASET_PATH, "config.json")
with open(config_path, 'r') as f:
    config = json.load(f)
dataset_structure = pd.read_csv(Path.joinpath(DATASET_PATH, config["scans_list_path"]))
dataset_img_dir_path = Path.joinpath(DATASET_PATH, config["relative_img_path"])

SCENE_MAP = {
    str(scene_id) : {
        "light_name" : scans.loc[scans['light'] == 1, 'name'].iloc[0].replace(".czi", ""),
        "dark_name" : scans.loc[scans['light'] == 0, 'name'].iloc[0].replace(".czi", ""),
        "env" : scans['environment'].iloc[0]
    }
    for scene_id, scans in dataset_structure.groupby("scene")
}

TARGET_H5_FILES = [ancient(f"results/roi/{scene_id}.h5") for scene_id in SCENE_MAP.keys()]

TARGET_MIRP_FEATURES = [ancient(f"results/features/mirp/{env}.csv") for env in config["classes"]]

TARGET_SEGMENTATION = [
        ancient(f"results/segmentation/cellpose/{SCENE_MAP[scene_id]["env"]}/{SCENE_MAP[scene_id]["light_name"]}.png")
        for scene_id in SCENE_MAP
    ]

TARGET_TOP_SELECTIONS = [ancient(f"results/features/top/selected_{n}.csv") for n in range(4)]

rule all:
    input: 
        segm = TARGET_SEGMENTATION,
        rois = TARGET_H5_FILES,
        mirp = TARGET_MIRP_FEATURES,
        fwzi = "results/features/featurewiz_selection.txt",
        # top_selections = TARGET_TOP_SELECTIONS

rule czi_to_np:
    input: str(dataset_img_dir_path) + "/{env}/{img}.czi"
    output: temp("results/interim/{env}/{img}.npy")
    conda: "envs/czi_env.yaml"
    shell: "python scripts/czi_img_loader.py --input {input} --output {output}"


rule cellpose_segmentation:
    input:
        "results/interim/{env}/{img}.npy"
    output:
        protected("results/segmentation/cellpose/{env}/{img}.png")
    conda:
        "envs/cellpose_env.yaml"
    shell:
        """
        python scripts/segmentation.py --input_path {input} --output_path {output}
        """


rule extract_roi: # process dark_images + masks into h5py files
    input: 
        # w for wildcards
        dark_img = lambda w: f"results/interim/{SCENE_MAP[w.scene_id]['env']}/{SCENE_MAP[w.scene_id]['dark_name']}.npy",
        light_mask = lambda w: ancient(f"results/segmentation/cellpose/{SCENE_MAP[w.scene_id]['env']}/{SCENE_MAP[w.scene_id]['light_name']}.png") # ancient - do not recreate if already exists (ignore old timestamp)
    output:
        "results/roi/{scene_id}.h5"
    conda:
        "envs/ml_vision.yaml"
    shell:
        """
        python scripts/extract_roi.py --image {input.dark_img} --mask {input.light_mask} --output {output}
        """

# rule extract_features:
#     input: # force snakemake to creat all files, but as input pass only directory
#         dataset_split_data = lambda wildcards: [ # w for wildcards
#                 ancient(f"results/segmentation/cellpose/{SCENE_MAP[str(scene_id)]['env']}/{SCENE_MAP[str(scene_id)]['light_name']}.png") # created segmentation of given scene
#                 for scene_id in config["dataset_split"][wildcards.dataset_split] # use snakemake wildecards to determine dataset_split
#             ]     
#     params: # actual inputs
#         segm_dir = "results/segmentation/cellpose"
#     output:
#         protected("results/features/mirp_{dataset_split}.csv")
#     conda:
#         "envs/mirp.yaml"
#     shell:
#         """
#         python scripts/mirp_extract.py --segm_dir {params.segm_dir} --dataset_split {wildcards.dataset_split} --output {output}
#         """

rule extract_features:
    input: # force snakemake to creat all files, but as input pass only directory
        complete_env_segm = lambda wildcards: [
                ancient(f"results/segmentation/cellpose/{SCENE_MAP[scene_id]['env']}/{SCENE_MAP[scene_id]['light_name']}.png")
                for scene_id in SCENE_MAP if SCENE_MAP[scene_id]["env"] == wildcards.env
            ] 
    params: # actual inputs
        segm_dir = "results/segmentation/cellpose/{env}"
    output:
        protected("results/features/mirp/{env}.csv")
    conda:
        "envs/mirp.yaml"
    shell:
        """
        python scripts/mirp_extract_by_env.py --segm_dir {params.segm_dir} --enviroment {wildcards.env} --output {output}
        """

rule filter_features:
    input:
        extracted_features = TARGET_MIRP_FEATURES
    params:
        extr_ftr_dir = "results/features/mirp"
    output:
        "results/features/featurewiz_selection.txt"
    conda:
        "envs/featurewiz.yaml"
    shell:
        """
        python scripts/run_featurewiz.py --extr_ftr_dir {params.extr_ftr_dir} --output {output}
        """

# rule find_best_n_features:
#     input:
#         selected_f = ancient("results/features/featurewiz_selection.txt"),
#         train_split = ancient("results/features/mirp_train.csv"),
#         val_split = ancient("results/features/mirp_val.csv")
#     output:
#         "results/features/top/selected_{n}.csv"
#     conda:
#         "envs/mirp.yaml"
#     shell:
#         """
#         python scripts/select_best_features.py \
#         --select_n_features {wildcards.n} \
#         --pre_selected_features {input.selected_f} \
#         --train_split {input.train_split} \
#         --val_split {input.val_split}\
#         --output {output}
#         """

rule find_best_n_features:
    input:
        selected_f = ancient("results/features/featurewiz_selection.txt"),
    params:
        extr_ftr_dir = "results/features/mirp"
    output:
        "results/features/top/selected_{n}.csv"
    conda:
        "envs/mirp.yaml"
    shell:
        """
        python scripts/select_best_features.py \
        --select_n_features {wildcards.n} \
        --pre_selected_features {input.selected_f} \
        --data_dir {params.extr_ftr_dir} \
        --output {output}
        """


# SNAKEMAKE COMANDS

# DRY RUN
# snakemake -n

# ACTUAL RUN
# snakemake --use-conda --quiet --cores 1
# snakemake --use-conda --cores 1
# snakemake --use-conda --cores 3

# SAVE DAG AS IMG
# snakemake --dag | dot -Tpng > workflow_dag.png


# CONDA ENVS
# snakemake --list-conda-envs
# environment     container       location
# envs/cellpose_env.yaml          .snakemake/conda/f0aac5aaed4209e36de94e118b849d05_
# envs/czi_env.yaml               .snakemake/conda/0aa85d447416ef846da9cedbe996cbc3_
# envs/featurewiz.yaml            .snakemake/conda/089e14863a2542433d6154648bf0f73e_
# envs/ml_vision.yaml             .snakemake/conda/7f69f82cf5e2d5fc9f5eb4294544276a_
# envs/mirp.yaml          .snakemake/conda/e8c059b11ba5ffa27872dfd97b236726_