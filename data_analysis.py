
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dataset_loader import Dataset
from dataclasses import dataclass, field
from collections import defaultdict
from config.config import DATASETS, DATASET_PATHS
from img_preprocess import stadard_preprocess

def extract_DAY_medium(label_name: str):
    a = label_name.split("_")
    # medium = "UNKNOWN"
    # for i in mediums:
    #     if i in label_name:
    #         medium = i
    #         break
    
    # day = "UNKNOWN"
    # for i in days:
    #     if i in label_name:
    #         day = i
            # break
    
    # return medium, day

    return a[-1], a[-2]


if __name__ == "__main__":
    # mediums = ["Acetate", "SD", "YPD"]
    # days = ["DAY_1", "DAY_3"]
    all_datasets = [
        DATASETS.YPD_SD_Acetate_DAY_1_Acetate, DATASETS.YPD_SD_Acetate_DAY_3_Acetate,
        DATASETS.YPD_SD_Acetate_DAY_1_SD,      DATASETS.YPD_SD_Acetate_DAY_3_SD,
        DATASETS.YPD_SD_Acetate_DAY_1_YPD,     DATASETS.YPD_SD_Acetate_DAY_3_YPD,
    ]


    datasets = [
        [DATASETS.YPD_SD_Acetate_DAY_1_Acetate, DATASETS.YPD_SD_Acetate_DAY_3_Acetate],
        [DATASETS.YPD_SD_Acetate_DAY_1_SD,      DATASETS.YPD_SD_Acetate_DAY_3_SD],
        [DATASETS.YPD_SD_Acetate_DAY_1_YPD,     DATASETS.YPD_SD_Acetate_DAY_3_YPD],
    ]

    dataset_intensities = dict()
    medium_intensities = {"Acetate" : [], "SD" : [], "YPD" : []}
    day_intensities = {"1" : [], "3" : []}
    

    for dataset in datasets:
        plt.figure(figsize=(10,6))
        for dataset_class in dataset:
            d = Dataset(DATASET_PATHS[dataset_class])

            label = dataset_class.name
            print(label)

            dataset_intensities[label] = []
            for i, scene in enumerate(d.scenes):
                print(f"{label}; {i+1}/{len(d.scenes)}")

                first_not_None_layer_id = next((i for i, x in enumerate(scene.dark) if x is not None), None)

                gray_img = scene.dark[first_not_None_layer_id]
                img_name = scene.dark_names[first_not_None_layer_id]


                for mask_index, mask in enumerate(scene.masks):
                    roi, mask_roi, denoised_roi = stadard_preprocess(gray_img, mask)

                    data = denoised_roi[mask_roi]

                    dataset_intensities[label].append(data)

            dataset_intensities[label] = np.concatenate(dataset_intensities[label])

            medium, day = extract_DAY_medium(label)
            medium_intensities[medium].append(dataset_intensities[label])
            day_intensities[day].append(dataset_intensities[label])



            
            hist, bins = np.histogram(dataset_intensities[label], bins=256, range=(0, 1))
            hist = hist / hist.sum() # normalize histogram by number of pixels pre class
            bin_width = bins[1] - bins[0]

            # plt.plot(hist, label=f"Class {label}")
            plt.bar(bins[:-1], hist, width=bin_width, alpha=0.5, label=f"Class {label}")

        plt.legend()
        plt.title(f"dataset {medium}")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        # plt.show()
        # plt.savefig(f"{label}_histogram.png")
        plt.savefig(f"visual/DA_histogram_dataset_{medium}.pdf") 
        plt.close()


    



    plt.figure(figsize=(10,6))
    for day, pixels in day_intensities.items():
        pixels = np.concatenate(pixels)
        hist, bins = np.histogram(pixels, bins=256, range=(0, 1))
        hist = hist / hist.sum() # normalize histogram by number of pixels pre class
        bin_width = bins[1] - bins[0]
        plt.bar(bins[:-1], hist, width=bin_width, alpha=0.5, label=f"day {day}")
    plt.legend()
    plt.title(f"DAY")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    # plt.show()
    plt.savefig(f"visual/DA_histogram_day.pdf") 
    plt.close()












    plt.figure(figsize=(10,6))
    for medium, pixels in medium_intensities.items():
        pixels = np.concatenate(pixels)
        hist, bins = np.histogram(pixels, bins=256, range=(0, 1))
        hist = hist / hist.sum() # normalize histogram by number of pixels pre class
        bin_width = bins[1] - bins[0]
        plt.bar(bins[:-1], hist, width=bin_width, alpha=0.5, label=f"{medium}")
    plt.legend()
    plt.title(f"MEDIUM")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    # plt.show()
    plt.savefig(f"visual/DA_histogram_medium.pdf") 
    plt.close()





