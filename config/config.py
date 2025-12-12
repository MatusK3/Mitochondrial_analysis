from enum import Enum
from pathlib import Path



feature_extraction_output_loaction = "outputs"


class DATASET_TYPES(Enum):
    OIR = 0
    SCI = 1

class DATASETS(Enum):
    OLYMPUS_TEST = 0
    SCI_TEST = 1
    YPD_Acetate_DAY_1_Acetate=2
    YPD_SD_Acetate_DAY_1_Acetate=3
    YPD_SD_Acetate_DAY_1_SD=4
    YPD_SD_Acetate_DAY_1_YPD=5
    YPD_SD_Acetate_DAY_3_Acetate=6
    YPD_SD_Acetate_DAY_3_SD=7
    YPD_SD_Acetate_DAY_3_YPD=8


DATASET_PATHS = {
    DATASETS.OLYMPUS_TEST : {
        "data_types" : DATASET_TYPES.OIR,
        "path" : Path("D:/Olympus_magMag234.85_2025-02-26/DiOC6"),
        "structure_path" : Path("C:/Work/Matfyz/Thesis/Mitochondrial_analysis/config/dataset_descriptions/OIR_TEST.csv"),
        "segmentation_masks_path" : None,
    },
    DATASETS.SCI_TEST : { # mitochondria
        "data_types" : DATASET_TYPES.SCI,
        "path" : Path("C:/Work/Matfyz/Thesis/data/zeiss_matfyz/magMag 234.85/DiOC6"),
        "structure_path" : Path("C:/Work/Matfyz/Thesis/Mitochondrial_analysis/config/dataset_descriptions/SCI_TEST.csv"),
        "segmentation_masks_path" : None,
    },


    DATASETS.YPD_Acetate_DAY_1_Acetate : { # core
        "data_types" : DATASET_TYPES.SCI,
        "path" : Path("C:/Work/Matfyz/Thesis/data/zeiss_matfyz/magMag 234.85/YPD_Acetate/Day 1/Acetate"),
        "structure_path" : Path(""),
        "segmentation_masks_path" : None,
    },



    DATASETS.YPD_SD_Acetate_DAY_1_Acetate : { # mitochondria
        "data_types" : DATASET_TYPES.SCI,
        "path" : Path("C:\Work\Matfyz\Thesis\data\zeiss_matfyz\magMag 234.85\YPD_SD_Acetate\Day 1\Acetate"),
        "structure_path" : Path("C:/Work/Matfyz/Thesis/Mitochondrial_analysis/config/dataset_descriptions/YPD_SD_Acetate_DAY_1_Acetate.csv"),
        "segmentation_masks_path" : Path("C:/Work/Matfyz/Thesis/data/zeiss_matfyz/magMag 234.85/annotations/YPD_SD_Acetate/DAY_1/Acetate"),
    },
    DATASETS.YPD_SD_Acetate_DAY_3_Acetate : { # mitochondria
        "data_types" : DATASET_TYPES.SCI,
        "path" : Path("C:\Work\Matfyz\Thesis\data\zeiss_matfyz\magMag 234.85\YPD_SD_Acetate\Day 3\Acetate"),
        "structure_path" : Path("C:/Work/Matfyz/Thesis/Mitochondrial_analysis/config/dataset_descriptions/YPD_SD_Acetate_DAY_3_Acetate.csv"),
        "segmentation_masks_path" : Path("C:/Work/Matfyz/Thesis/data/zeiss_matfyz/magMag 234.85/annotations/YPD_SD_Acetate/DAY_3/Acetate"),
    },


    DATASETS.YPD_SD_Acetate_DAY_1_SD : { # mitochondria
        "data_types" : DATASET_TYPES.SCI,
        "path" : Path("C:\Work\Matfyz\Thesis\data\zeiss_matfyz\magMag 234.85\YPD_SD_Acetate\Day 1\SD"),
        "structure_path" : Path("C:/Work/Matfyz/Thesis/Mitochondrial_analysis/config/dataset_descriptions/YPD_SD_Acetate_DAY_1_SD.csv"),
        "segmentation_masks_path" : Path("C:/Work/Matfyz/Thesis/data/zeiss_matfyz/magMag 234.85/annotations/YPD_SD_Acetate/DAY_1/SD"),
    },
    DATASETS.YPD_SD_Acetate_DAY_3_SD : { # mitochondria
        "data_types" : DATASET_TYPES.SCI,
        "path" : Path("C:\Work\Matfyz\Thesis\data\zeiss_matfyz\magMag 234.85\YPD_SD_Acetate\Day 3\SD"),
        "structure_path" : Path("C:/Work/Matfyz/Thesis/Mitochondrial_analysis/config/dataset_descriptions/YPD_SD_Acetate_DAY_3_SD.csv"),
        "segmentation_masks_path" : Path("C:/Work/Matfyz/Thesis/data/zeiss_matfyz/magMag 234.85/annotations/YPD_SD_Acetate/DAY_3/SD"),
    },

    DATASETS.YPD_SD_Acetate_DAY_1_YPD : { # mitochondria
        "data_types" : DATASET_TYPES.SCI,
        "path" : Path("C:\Work\Matfyz\Thesis\data\zeiss_matfyz\magMag 234.85\YPD_SD_Acetate\Day 1\YPD"),
        "structure_path" : Path("C:/Work/Matfyz/Thesis/Mitochondrial_analysis/config/dataset_descriptions/YPD_SD_Acetate_DAY_1_YPD.csv"),
        "segmentation_masks_path" : Path("C:/Work/Matfyz/Thesis/data/zeiss_matfyz/magMag 234.85/annotations/YPD_SD_Acetate/DAY_1/YPD"),
    },
    DATASETS.YPD_SD_Acetate_DAY_3_YPD : { # mitochondria
        "data_types" : DATASET_TYPES.SCI,
        "path" : Path("C:\Work\Matfyz\Thesis\data\zeiss_matfyz\magMag 234.85\YPD_SD_Acetate\Day 3\YPD"),
        "structure_path" : Path("C:/Work/Matfyz/Thesis/Mitochondrial_analysis/config/dataset_descriptions/YPD_SD_Acetate_DAY_3_YPD.csv"),
        "segmentation_masks_path" : Path("C:/Work/Matfyz/Thesis/data/zeiss_matfyz/magMag 234.85/annotations/YPD_SD_Acetate/DAY_3/YPD"),
    },


}

