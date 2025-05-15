import numpy as np
from numpy import ndarray
from pathlib import Path
import cv2 as cv
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
from aicspylibczi import CziFile
import pandas as pd

from segmentation_classic import segmentate
from config.config import DATASET_TYPES

class Scene(ABC):
    @abstractmethod
    def __init__(self, data_path :Path, light_name :str, dark_name :str, scene_id :int):
        # super().__init__()
        ...

    def show_scene(self):
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        # Original image
        axes[0,0].imshow(self.light, cmap='gray')
        axes[0,0].set_title('light')
        
        # mask
        axes[0,1].axis("off")
        axes[0,2].axis("off")
        if self.mask is not None:
            axes[0,1].imshow(self.mask, cmap='gray')
            axes[0,1].set_title('mask')
            axes[0,1].axis("on")
        
        # dark layers
        for i in range(3):
            if self.dark[i] is not None:
                axes[1,i].imshow(self.dark[i], cmap='gray')
                axes[1,i].set_title(f'layer {i}')
    
        plt.tight_layout()
        plt.show()


class OIR_Scene(Scene):
    # def __init__(self, data_path :Path, light_name :str, dark_name :str, scene_id :int):
    #     self.scene_id = scene_id
    #     self.light_name :ndarray = light_name
    #     self.dark_name :ndarray = dark_name
    #     self.light_path = data_path.joinpath(light_name)
    #     self.dark_path = data_path.joinpath(dark_name)
    #     self.light = cv.imread(str(self.light_path))
    #     self.dark = cv.imread(str(self.dark_path))

    #     self.mask = segmentate(self.light)
    #     # self.show_scene()



    def __init__(self, scans :pd.DataFrame):
        self.scene_id = scans['scene'].iloc[0]
        
        light_on = scans.loc[scans['light'] == 1]
        light_off = scans.loc[scans['light'] == 0]

    
class SCI_Scene(Scene):
    @staticmethod
    def _get_img(sci_file) -> ndarray:
        img_block, dims_list = sci_file.read_image(S=0, Z=0)  
        img_squeezed = np.squeeze(img_block)
        return img_squeezed


    def __init__(self, data_path :Path, scans :pd.DataFrame):
        self.scene_id = scans['scene'].iloc[0]

        self.light_name :str= scans.loc[(scans['light'] == 1, 'name')].values[0]
        self.light_path = data_path.joinpath(self.light_name)
        self.sci_light = CziFile(self.light_path)
        self.light = SCI_Scene._get_img(self.sci_light)

        self.mask = np.ones(self.light.shape, dtype=np.uint8) * 255
        
        self.dark_names = []
        self.dark_paths = []
        self.sci_dark = []
        self.dark = []
        
        light_off = scans.loc[scans['light'] == 0]
        for i in range(3):
            layer_record = light_off.loc[light_off['layer'] == i+1]
            if layer_record.empty:
                self.dark_names.append(None)
                self.dark_paths.append(None)
                self.sci_dark.append(None)
                self.dark.append(None)
            else:
                layer_name = layer_record['name'].values[0]
                layer_path = data_path.joinpath(layer_name)
                sci_layer = CziFile(layer_path)
                layer_scan = SCI_Scene._get_img(sci_layer)

                self.dark_names.append(layer_name)
                self.dark_paths.append(layer_path)
                self.sci_dark.append(sci_layer)
                self.dark.append(layer_scan)
            
            


def get_scene(dataset_type :DATASET_TYPES, data_path :Path, scans :pd.DataFrame) -> Scene:
    if(dataset_type == DATASET_TYPES.OIR): return OIR_Scene(data_path, scans)
    if(dataset_type == DATASET_TYPES.SCI): return SCI_Scene(data_path, scans)
