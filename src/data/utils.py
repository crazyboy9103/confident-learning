from dataclasses import dataclass
from enum import Enum

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

@dataclass
class OverlookNoiseConfig:
    prob: float = 0.05

@dataclass
class BadLocNoiseConfig:
    prob: float = 0.05
    max_pixel: int = 50

@dataclass
class SwapNoiseConfig:
    num_classes_to_swap: int = 3
    prob: float = 0.05

class NoiseType(Enum):
    NORMAL: int = 0
    OVERLOOK: int = 1
    BADLOC: int = 2
    SWAP: int = 3
class SubsetWithTransform(Dataset):
    def __init__(self, dataset, idxs, transform=None):
        self.dataset = dataset
        self.idxs = idxs
        self.transform = transform
    
    # We want to iterate too
    def __iter__(self):
        for idx in self.idxs:
            image, target = self.dataset[idx]
            if self.transform:
                image, target = self.transform(image, target)

            yield image, target
                        
    def __getitem__(self, index):
        image, target = self.dataset[self.idxs[index]]
        if self.transform:
            image, target = self.transform(image, target)
        return image, target
        
    def __len__(self):
        return len(self.idxs)

def polygon_area(polygon):
    reshaped_polygon = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    return cv2.contourArea(reshaped_polygon)

def collate_fn(batch):
    return tuple(zip(*batch))

def segmentation_to_mask(segmentation, *, canvas_size):
    from pycocotools import mask

    segmentation = (
        mask.frPyObjects(segmentation, *canvas_size)
        if isinstance(segmentation, dict)
        else mask.merge(mask.frPyObjects(segmentation, *canvas_size))
    )
    return torch.from_numpy(mask.decode(segmentation))