import numpy as np
from numpy import ndarray
from pathlib import Path
import cv2 as cv
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
from aicspylibczi import CziFile
import pandas as pd

import matplotlib.colors as mcolors
import matplotlib.cm as cm
import random

from config.config import DATASET_TYPES

class Scene(ABC):
    @abstractmethod
    def __init__(self, data_path :Path, light_name :str, dark_name :str, scene_id :int):
        # super().__init__()
        ...


    def show_scene(self):
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))

        fig.suptitle(f"name: {self.light_name}, scene_id: {self.scene_id}", fontsize=16)


        # Original image
        axes[0,0].imshow(self.light, cmap='gray')
        axes[0,0].set_title(f'light, {self.light_name}')
        
        # mask
        axes[0,1].axis("off")
        if self.masks is not None:
            axes[0,1].imshow(self.light, cmap='gray') # background image
            for mask in self.masks:
                color = np.array(mcolors.to_rgba(cm.tab20(random.randint(0, 19)), alpha=0.6))
                # Create transparent overlay
                overlay = np.zeros((*mask.shape, 4))
                overlay[mask] = color
                axes[0,1].imshow(overlay)
                # plt.contour(mask, colors='k', linewidths=2)
            axes[0,1].set_title('mask')
            axes[0,1].axis("on")

            axes[1,1].imshow(np.sum(self.masks, axis=0), cmap='gray')
            axes[0,1].set_title('boolean masks')
            axes[0,1].axis("on")

        
        # first dark layers
        for i in range(3):
            if self.dark[i] is not None:
                axes[1,0].imshow(self.dark[i], cmap='gray')
                axes[1,0].set_title(f'layer {i}, {self.dark_names[i]}')
                break

        
    
        plt.tight_layout()
        plt.show()

    def save_to_png(self, dir_path, light=True, dark=False):
        name = self.light_name.split(".")[0]
        
        if light:
            location = dir_path + "/" + name + ".png"


            x = self.light.astype(np.float32)
            x -= np.min(x)
            x /= np.max(x)
            x *= 255
            x = np.uint8(x)
            cv.imwrite(location, x)

        if dark:
            for i in range(3):
                if self.dark[i] is not None:
                    location = dir_path + "/" + name + "_dark_"+ i + ".png"
                    cv.imwrite(location, self.dark[i])


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


    def __init__(self, data_path :Path, scans :pd.DataFrame, segmentation_path :Path):
        self.scene_id = scans['scene'].iloc[0]

        # load light image
        self.light_name :str= scans.loc[(scans['light'] == 1, 'name')].values[0]
        self.light_path = data_path.joinpath(self.light_name)
        self.sci_light = CziFile(self.light_path)
        self.light = SCI_Scene._get_img(self.sci_light)
        
        self.dark_names = []
        self.dark_paths = []
        self.sci_dark = []
        self.dark = []
        
        # load dark images
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
        
        # load segmentation masks
        if segmentation_path is not None:
            segmentation_file_name = f'{self.light_name.split(".")[0]}.npy'
            masks_file_path =  segmentation_path.joinpath(segmentation_file_name)
            self.masks = np.load(masks_file_path, allow_pickle=True)
        else:
            self.masks = None

            
            


def get_scene(dataset_type :DATASET_TYPES, data_path :Path, scans :pd.DataFrame, segmentation_path :Path = None) -> Scene:
    if(dataset_type == DATASET_TYPES.OIR): return OIR_Scene(data_path, scans, segmentation_path)
    if(dataset_type == DATASET_TYPES.SCI): return SCI_Scene(data_path, scans, segmentation_path)
