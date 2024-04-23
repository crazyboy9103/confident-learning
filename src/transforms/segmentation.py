# Adapted from torchvision/references/segmentation
from typing import Callable

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2 as Tv2
from torchvision.transforms import functional as F
from torchvision.models.segmentation.fcn import FCN_ResNet50_Weights 

class PadIfSmaller(Tv2.Transform):
    def __init__(self, size, fill=0):
        super().__init__()
        self.size = size
        self.fill = Tv2._utils._setup_fill_arg(fill)

    def _get_params(self, sample):
        _, height, width = Tv2._utils.query_chw(sample)
        padding = [0, 0, max(self.size - width, 0), max(self.size - height, 0)]
        needs_padding = any(padding)
        return dict(padding=padding, needs_padding=needs_padding)

    def _transform(self, inpt, params):
        if not params["needs_padding"]:
            return inpt

        fill = Tv2._utils._get_fill(self.fill, type(inpt))
        fill = Tv2._utils._convert_fill_arg(fill)

        return Tv2.functional.pad(inpt, padding=params["padding"], fill=fill)

class SegmentationPresetTrain:
    def __init__(
        self,
        *,
        base_size,
        crop_size,
        hflip_prob=0.5,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        transforms = [
            Tv2.ToImage(), 
            Tv2.RandomResize(min_size=int(0.5 * base_size), max_size=int(2.0 * base_size))
        ]

        if hflip_prob > 0:
            transforms += [Tv2.RandomHorizontalFlip(hflip_prob)]

        # We need a custom pad transform here, since the padding we want to perform here is fundamentally
        # different from the padding in `RandomCrop` if `pad_if_needed=True`.
        transforms += [
            PadIfSmaller(crop_size, fill={tv_tensors.Mask: 255, "others": 0}), 
            Tv2.RandomCrop(crop_size),
            Tv2.ToDtype(dtype={tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64, "others": None}, scale=True),
            Tv2.Normalize(mean=mean, std=std),
            Tv2.ToPureTensor()
        ]

        self.transforms = Tv2.Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(
        self, *, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
    ):
        transforms = [
            Tv2.ToImage(), 
            Tv2.Resize(size=(base_size, base_size)),
            Tv2.ToDtype(torch.float, scale=True),
            Tv2.Normalize(mean=mean, std=std),
            Tv2.ToPureTensor()
        ]
        self.transforms = Tv2.Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)
    
def fcn_transform() -> Callable:
    """
    Get transform function for FCN model
    :return: transform function for input images
    """
    weights = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    return weights.transforms()

def get_transform(is_train, **kwargs):
    if is_train:
        return SegmentationPresetTrain(
            **kwargs
        )
    
    return SegmentationPresetEval(**kwargs)