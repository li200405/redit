from typing import Dict, Type
from .ChongqingDataset import ChongqingDataset
from .PASTISDataset import PASTISDataset

DATASETS = {
    "chongqing": ChongqingDataset,
    "pastis": PASTISDataset,
}
