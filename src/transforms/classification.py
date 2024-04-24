# Adapted from torchvision/references/classification
from typing import Callable

import torch
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode
from torchvision.models.efficientnet import EfficientNet_B0_Weights

class ClassificationPresetTrain:
    def __init__(
        self,
        *,
        resize_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        ra_magnitude=9,
        augmix_severity=3,
        random_erase_prob=0.0,
    ):
        transforms = [
            v2.PILToTensor(), 
            v2.Resize(resize_size, interpolation=interpolation, antialias=True),
            # v2.RandomResizedCrop(crop_size, interpolation=interpolation, antialias=True),
        ]
        if hflip_prob > 0:
            transforms.append(v2.RandomHorizontalFlip(hflip_prob))

        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                transforms.append(v2.RandAugment(interpolation=interpolation, magnitude=ra_magnitude))

            elif auto_augment_policy == "ta_wide":
                transforms.append(v2.TrivialAugmentWide(interpolation=interpolation))

            elif auto_augment_policy == "augmix":
                transforms.append(v2.AugMix(interpolation=interpolation, severity=augmix_severity))

            else:
                aa_policy = v2.AutoAugmentPolicy(auto_augment_policy)
                transforms.append(v2.AutoAugment(policy=aa_policy, interpolation=interpolation))

        transforms += [
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
        
        if random_erase_prob > 0:
            transforms.append(v2.RandomErasing(p=random_erase_prob))

        transforms.append(v2.ToPureTensor())

        self.transforms = v2.Compose(transforms)

    def __call__(self, image, target):
        return self.transforms(image), target 

class ClassificationPresetEval:
    def __init__(
        self,
        *,
        resize_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
    ):
        transforms = [
            v2.PILToTensor(),
            v2.Resize(resize_size, interpolation=interpolation, antialias=True),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(mean=mean, std=std),
            v2.ToPureTensor()
        ]

        self.transforms = v2.Compose(transforms)

    def __call__(self, image, target):
        return self.transforms(image), target

def efficientnetb0_transform() -> Callable:
    """
    Get transform function for EfficientNet model with specified model name
    :return: transform function for input images
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    return weights.transforms()

def get_transform(is_train, **kwargs):
    if is_train:
        return ClassificationPresetTrain(
            **kwargs
        )
    
    return ClassificationPresetEval(**kwargs)
    