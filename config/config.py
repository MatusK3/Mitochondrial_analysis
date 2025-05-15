from enum import Enum
from pathlib import Path

class DATASET_TYPES(Enum):
    OIR = 0
    SCI = 1

class DATASETS(Enum):
    OLYMPUS_TEST = 0
    SCI_TEST = 1


DATASET_PATHS = {
    DATASETS.OLYMPUS_TEST : {
        "data_types" : DATASET_TYPES.OIR,
        "path" : Path("D:/Olympus_magMag234.85_2025-02-26/DiOC6"),
        "structure_path" : Path("C:/Work/Matfyz/Thesis/Mitochondrial_analysis/config/OIR_TEST_dataset_descibtion.csv"),
    },
    DATASETS.SCI_TEST : {
        "data_types" : DATASET_TYPES.SCI,
        "path" : Path("C:/Work/Matfyz/Thesis/data/zeiss_matfyz/magMag 234.85/DiOC6"),
        "structure_path" : Path("C:/Work/Matfyz/Thesis/Mitochondrial_analysis/config/SCI_TEST_dataset_descibtion.csv"),
    }
}

