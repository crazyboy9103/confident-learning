# Adapted from torchvision/references/segmentation
from typing import Callable

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.models.segmentation.fcn import FCN_ResNet50_Weights 

class PadIfSmaller(v2.Transform):
    # Ensures the image is at least crop_size x crop_size by padding 
    def __init__(self, size, fill=0):
        super().__init__()
        self.size = size
        self.fill = v2._utils._setup_fill_arg(fill)

    def _get_params(self, sample):
        _, height, width = v2._utils.query_chw(sample)
        padding = [0, 0, max(self.size - width, 0), max(self.size - height, 0)]
        needs_padding = any(padding)
        return dict(padding=padding, needs_padding=needs_padding)

    def _transform(self, inpt, params):
        if not params["needs_padding"]:
            return inpt

        fill = v2._utils._get_fill(self.fill, type(inpt))
        fill = v2._utils._convert_fill_arg(fill)

        return v2.functional.pad(inpt, padding=params["padding"], fill=fill)

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
            v2.ToImage(), 
            # antialias=True is future default in torchvision
            v2.RandomResize(min_size=int(0.5 * base_size), max_size=int(2.0 * base_size), antialias=True)
        ]

        if hflip_prob > 0:
            transforms += [v2.RandomHorizontalFlip(hflip_prob)]

        # We need a custom pad transform here, since the padding we want to perform here is fundamentally
        # different from the padding in `RandomCrop` if `pad_if_needed=True`.
        transforms += [
            PadIfSmaller(crop_size, fill={tv_tensors.Mask: 255, "others": 0}), # padding masks with 255, which is ignored in training loss computation
                                                                               # and images with zeros. 
            v2.RandomCrop(crop_size),
            v2.ToDtype(dtype={tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64, "others": None}, scale=True),
            v2.Normalize(mean=mean, std=std),
            v2.ToPureTensor()
        ]

        self.transforms = v2.Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(
        self, *, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
    ):
        transforms = [
            v2.ToImage(), 
            # v2.Resize(size=(base_size, base_size)),
            # v2.ToDtype(torch.float, scale=True),
            v2.ToDtype(dtype={tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64, "others": None}, scale=True),
            v2.Normalize(mean=mean, std=std),
            v2.ToPureTensor()
        ]
        self.transforms = v2.Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)
    
def fcn_transform() -> Callable:
    """
    Get transform function for FCN model
    :return: transform function for input images
    """
    weights = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    return weights.transforms()