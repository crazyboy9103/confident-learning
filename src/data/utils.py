from dataclasses import dataclass
from enum import Enum

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
@dataclass
class OverlookNoiseConfig:
    prob: float = 0.1

@dataclass
class BadLocNoiseConfig:
    prob: float = 0.1
    max_pixel: int = 50

@dataclass
class SwapNoiseConfig:
    num_classes_to_swap: int = 3
    prob: float = 0.1

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

def uniform_size_collate_fn(batch):
    images, labels = zip(*batch)
    
    max_height = max(img.shape[1] for img in images)
    max_width = max(img.shape[2] for img in images)

    if all(img.shape[1:] == (max_height, max_width) for img in images):
        return batch
    
    # Assuming all images have the same number of channels
    channel = images[0].shape[0]

    padded_images = torch.zeros((len(images), channel, max_height, max_width),
                                dtype=images[0].dtype, device=images[0].device)

    for i, img in enumerate(images):
        padding_left, add_to_right = divmod(max_width - img.shape[2], 2)
        padding_right = padding_left + add_to_right
        padding_top, add_to_bottom = divmod(max_height - img.shape[1], 2)
        padding_bottom = padding_top + add_to_bottom

        padded_images[i] = F.pad(img, (padding_left, padding_top, padding_right, padding_bottom), padding_mode='reflect')

    labels = torch.tensor(labels, dtype=torch.long, device=images[0].device)
    return padded_images, labels
    
def segmentation_to_mask(segmentation, *, canvas_size):
    from pycocotools import mask

    segmentation = (
        mask.frPyObjects(segmentation, *canvas_size)
        if isinstance(segmentation, dict)
        else mask.merge(mask.frPyObjects(segmentation, *canvas_size))
    )
    return torch.from_numpy(mask.decode(segmentation))