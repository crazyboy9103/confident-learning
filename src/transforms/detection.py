# Adapted from torchvision/references/detection
from typing import Callable

import torch
from torchvision.transforms import v2
from torchvision.models.detection.retinanet import RetinaNet_ResNet50_FPN_Weights

class DetectionPresetTrain:
    def __init__(
        self,
        *,
        data_augmentation,
        hflip_prob=0.5,
    ):
        transforms = [
            v2.ToImage(),
        ]

        if data_augmentation == "hflip":
            transforms += [v2.RandomHorizontalFlip(p=hflip_prob)]

        elif data_augmentation == "multiscale":
            transforms += [
                v2.RandomShortestSize(min_size=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800), max_size=1333),
                v2.RandomHorizontalFlip(p=hflip_prob),
            ]

        else:
            raise ValueError(f'Unknown data augmentation policy "{data_augmentation}"')

        transforms += [
            v2.ToDtype(torch.float, scale=True),
            # v2.ConvertBoundingBoxFormat(tv_tensors.BoundingBoxFormat.XYXY),
            v2.SanitizeBoundingBoxes(),
            v2.ToPureTensor(),
        ]

        self.transforms = v2.Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)


class DetectionPresetEval:
    def __init__(self):
        transforms = [
            v2.ToImage(),
            v2.ToDtype(torch.float, scale=True),
            v2.SanitizeBoundingBoxes(),
            v2.ToPureTensor()
        ]

        self.transforms = v2.Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)

def retinanet_transform() -> Callable:
    """
    Get transform function for RetinaNet model
    :return: transform function for input images
    """
    weights = RetinaNet_ResNet50_FPN_Weights.COCO_V1
    return weights.transforms()

def get_transform(is_train, **kwargs):
    if is_train:
        return DetectionPresetTrain(
            **kwargs
        )
    
    return DetectionPresetEval(**kwargs)
    