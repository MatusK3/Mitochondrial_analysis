import numpy as np
from numpy import ndarray
from pathlib import Path
import pandas as pd

from scene import Scene, get_scene
from config.config import DATASET_TYPES, DATASET_PATHS, DATASETS


class Dataset():
    def __init__(self, dataset_paths :dict):
        self.scenes :list[Scene] = []
        self.load_data(dataset_paths["path"], dataset_paths["structure_path"], dataset_paths["data_types"])

    
    def load_data(self, data_path: Path, dataset_structure_path: Path, data_type :DATASET_TYPES):
        self.data_path = data_path
        self.dataset_structure_path = dataset_structure_path

        df = pd.read_csv(dataset_structure_path)
        self.scenes :list[Scene] = []
        

        for scene_id in df['scene'].unique():
            scans = df.loc[(df['scene'] == scene_id)]
            self.scenes.append(get_scene(data_type, self.data_path, scans))


if __name__ =="__main__":
#    d = Dataset(DATASET_PATHS[DATASETS.OLYMPUS_TEST])
   d = Dataset(DATASET_PATHS[DATASETS.SCI_TEST])

   for scene in d.scenes:
       scene.show_scene()




