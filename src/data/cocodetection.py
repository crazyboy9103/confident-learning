from typing import Any, Dict, Optional, Callable, Literal, Union
import random
from itertools import combinations

import numpy as np
import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.datasets import CocoDetection
from torchvision.transforms.v2 import functional as F
from torchvision.tv_tensors._dataset_wrapper import list_of_dicts_to_dict_of_lists
from sklearn.model_selection import KFold

from .utils import SubsetWithTransform
from .utils import collate_fn, polygon_area, segmentation_to_mask
from .utils import OverlookNoiseConfig, BadLocNoiseConfig, SwapNoiseConfig, NoiseType

class NoisyCocoDetection(CocoDetection):
    def __getitem__(self, index: int):
        # Refer to https://pytorch.org/vision/stable/generated/torchvision.datasets.wrap_dataset_for_transforms_v2.html
        # for the explanation of the following code
        image_id = self.ids[index]
        image = self._load_image(image_id)
        target = self._load_target(image_id)

        canvas_size = tuple(F.get_size(image))

        # If the target is empty, create an empty target
        if not target:
            target = {"image_id": image_id}
            if self.task == "seg":
                # For semantic segmentation, if the target is empty, we need to create an empty mask, i.e. a tensor of zeros
                target["masks"] = tv_tensors.Mask(
                    torch.zeros(1, *canvas_size, dtype=torch.int64)
                )
            target["noisy_labels"] = torch.tensor([False,], dtype=torch.bool)
            target["noisy_label_types"] = torch.tensor([NoiseType.NORMAL.value,], dtype=torch.int64)
            return image, target
        
        batched_target = list_of_dicts_to_dict_of_lists(target)
        target = {"image_id": image_id}

        # Overlook must be dealt with as follows:
        # If the noise type is overlook, it is possible that we remove all the labels from an image
        # In such case, we need to return an empty mask or empty boxes, as required by the loss function
        if self.type == "overlook":
            # If all the labels are overlooked, return a mask of zeros for seg or empty boxes for det
            if all(batched_target["noisy_label"]):
                target["noisy_labels"] = torch.tensor(batched_target["noisy_label"])
                target["noisy_label_types"] = torch.tensor(batched_target["noisy_label_type"])
                if self.task == "seg":
                    target["masks"] = tv_tensors.Mask(
                        torch.zeros(1, *canvas_size, dtype=torch.int64)
                    )
                
                elif self.task == "det":
                    target["boxes"] = tv_tensors.BoundingBoxes(
                        torch.zeros(0, 4, dtype=torch.float32),
                        format=tv_tensors.BoundingBoxFormat.XYWH,
                        canvas_size=canvas_size
                    )
                target["labels"] = torch.tensor([], dtype=torch.int64)
                return image, target

            # If only some of the labels are overlooked, filter out the overlooked labels
            if self.task == "seg":
                batched_target["segmentation"] = [segmentation for segmentation, noisy_label in zip(batched_target["segmentation"], batched_target["noisy_label"]) if not noisy_label]

            elif self.task == "det":
                batched_target["bbox"] = [bbox for bbox, noisy_label in zip(batched_target["bbox"], batched_target["noisy_label"]) if not noisy_label]
        
            batched_target["category_id"] = [category_id for category_id, noisy_label in zip(batched_target["category_id"], batched_target["noisy_label"]) if not noisy_label]

        if self.task == "det":
            target["boxes"] = tv_tensors.BoundingBoxes(
                batched_target["bbox"],
                format=tv_tensors.BoundingBoxFormat.XYWH,
                canvas_size=canvas_size
            )

        # segmentation_to_mask(segmentation, canvas_size=canvas_size) is a binary mask of shape (height, width)
        # We multiply it by category_id to get a mask of shape (height, width) with the category_id as the pixel value
        # For semantic segmentation, we want the mask to be of shape (1, height, width) with the pixel value as the category_id
        # Therefore, we stack the masks and take the maximum value along the 0th dimension to get the final mask 
        # Neuro-T uses minimum value, but here we assume that there is no overlap between the masks (though there might be incorrect cases) 
        # so it does not matter whether we use minimum or maximum
        # The reason why we do not use minimum is that we want to keep 255 which is ignored in the loss function
        elif self.task == "seg":
            target["masks"] = tv_tensors.Mask(
                torch.stack(
                    [
                        segmentation_to_mask(segmentation, canvas_size=canvas_size) * category_id
                        for segmentation, category_id in zip(batched_target["segmentation"], batched_target["category_id"])
                    ]
                ).max(dim=0, keepdim=True).values
            )

        target["labels"] = torch.tensor(batched_target["category_id"])
        target["noisy_labels"] = torch.tensor(batched_target["noisy_label"])
        target["noisy_label_types"] = torch.tensor(batched_target["noisy_label_type"])
        return image, target
    
    def __init__(self, type: Literal["overlook", "badloc", "swap"], config: Union[OverlookNoiseConfig, BadLocNoiseConfig, SwapNoiseConfig], task, *args, **kwargs):
        super().__init__(*args, **kwargs)
        random.seed(42)
        # Synthesizing noisy labels from the original labels requires the following steps:
        #  Overlook (See OverlookNoiseConfig):
        #  - Randomly sample the annotations to remove for given prob
        #  - For each sampled annotation, add a flag to the annotation to track that it is noisy
        #  - Refer to the __getitem__ method for how to handle the case when all the labels are overlooked
        #  BadLoc (See BadLocNoiseConfig):
        #  - Randomly sample the annotations to perturb for given prob
        #  - For each sampled annotation, randomly sample the pixels to move the bounding box by a random amount dx, dy
        #  - Perturb the positions of the bounding box/segmentation by dx, dy
        #  - Clip the bounding box/segmentation to the image
        #  Swap (See SwapNoiseConfig):
        #  - Randomly sample the classes to swap for given num_classes_to_swap
        #  - Create pairs of classes to swap
        #  - For each pair, randomly sample the annotations to swap for given prob
        #  - Swap the classes of the sampled annotations
        self.task = task
        self.type = type
        self.config = config
        self.classes = self.coco.cats.copy()

        # Add a flag to each annotation to track whether it is noisy or not
        for ann_id, ann in self.coco.anns.items():
            ann["noisy_label"] = False
            ann["noisy_label_type"] =  NoiseType.NORMAL.value

        cat_ids = self.coco.getCatIds()
        ann_ids = self.coco.getAnnIds()
        
        match type:
            case "overlook":
                assert isinstance(config, OverlookNoiseConfig)
                num_noise = int(len(ann_ids) * config.prob)
                ann_ids_overlook = random.sample(ann_ids, num_noise)

                for ann_id in ann_ids_overlook:
                    self.coco.anns[ann_id]["noisy_label"] = True
                    self.coco.anns[ann_id]["noisy_label_type"] = NoiseType.OVERLOOK.value

            case "badloc":
                assert isinstance(config, BadLocNoiseConfig)
                num_noise = int(len(ann_ids) * config.prob)
                ann_ids_badloc = random.sample(ann_ids, num_noise)

                for ann_id in ann_ids_badloc:
                    self.coco.anns[ann_id]["noisy_label"] = True
                    self.coco.anns[ann_id]["noisy_label_type"] =  NoiseType.BADLOC.value

                    ann = self.coco.anns[ann_id]
                    
                    x1, y1, w, h = ann["bbox"]
                    x2, y2 = x1 + w, y1 + h

                    segmentation = ann["segmentation"][0] # zeroth element is the segmentation

                    # Fetch width and height for clipping within the image
                    img = self.coco.imgs[ann["image_id"]]
                    img_width, img_height = img["width"], img["height"]

                    # Pixels to move the bounding box 
                    dx = random.randint(-config.max_pixel, config.max_pixel)
                    dy = random.randint(-config.max_pixel, config.max_pixel)

                    x1 = x1 + dx
                    y1 = y1 + dy
                    x2 = x2 + dx
                    y2 = y2 + dy

                    # Clip the bounding box to the image
                    x1 = np.clip(x1, 0, img_width-1)
                    y1 = np.clip(y1, 0, img_height-1)
                    x2 = np.clip(x2, 0, img_width-1)
                    y2 = np.clip(y2, 0, img_height-1)

                    w = x2 - x1
                    h = y2 - y1

                    ann["bbox"] = [x1, y1, w, h]

                    # Move the segmentation
                    segmentation = np.array(segmentation).reshape(-1, 2)
                    segmentation[:, 0] = np.clip(segmentation[:, 0] + dx, 0, img_width-1)
                    segmentation[:, 1] = np.clip(segmentation[:, 1] + dy, 0, img_height-1)
                    ann["segmentation"] = [segmentation.flatten().tolist()]
                    ann["area"] = polygon_area(ann["segmentation"])

            case "swap":
                assert isinstance(config, SwapNoiseConfig)
                classes_to_swap = random.sample(cat_ids, config.num_classes_to_swap)
                pairs_to_swap = combinations(classes_to_swap, 2)
                
                ann_ids_class = {cat_id: self.coco.getAnnIds(catIds=cat_id) for cat_id in classes_to_swap}
                counts = {cat_id: len(ann_ids) for cat_id, ann_ids in ann_ids_class.items()}

                for i, j in pairs_to_swap:
                    ann_ids_i = ann_ids_class[i]
                    ann_ids_j = ann_ids_class[j]

                    num_to_swap_i = int(min(counts[i], len(ann_ids_class[i])) * config.prob)
                    num_to_swap_j = int(min(counts[j], len(ann_ids_class[j])) * config.prob)

                    ann_ids_swap_i = random.sample(ann_ids_i, num_to_swap_i)
                    ann_ids_swap_j = random.sample(ann_ids_j, num_to_swap_j)
                    
                    for ann_id in ann_ids_swap_i:
                        self.coco.anns[ann_id]["noisy_label"] = True
                        self.coco.anns[ann_id]["noisy_label_type"] = NoiseType.SWAP.value

                        ann = self.coco.anns[ann_id]
                        ann["category_id"] = j
                    
                    for ann_id in ann_ids_swap_j:
                        self.coco.anns[ann_id]["noisy_label"] = True
                        self.coco.anns[ann_id]["noisy_label_type"] = NoiseType.SWAP.value

                        ann = self.coco.anns[ann_id]
                        ann["category_id"] = i
class NoisyCocoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        annFile: str,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_transform: Callable = None, # for image augmentation Callable[PIL.Image]
        valid_transform: Callable = None, # for image augmentation Callable[PIL.Image]
        test_transform: Callable = None, # for image augmentation Callable[PIL.Image]
        task: Literal["det", "seg"] = "det",
        noise_type: Literal["overlook", "badloc", "swap"] = "overlook",
        noise_config: Union[OverlookNoiseConfig, BadLocNoiseConfig, SwapNoiseConfig] = OverlookNoiseConfig(),
        fold_index: int = 0, # for k-fold cross validation
        num_folds: int = 4, # for k-fold cross validation
    ) -> None:
        """NoisyCocoDataModule is a wrapper around torchvision `CocoDetection` dataset that adds noise to the labels. `root` and `annFile` are the arguments to the `CocoDetection` class. 
        Specifically, it can add three types of noise: `overlook`, `badloc`, and `swap`.
        1. `overlook`: Randomly remove some of the labels.
        2. `badloc`: Randomly move the bounding boxes or segmentations.
        3. `swap`: Randomly swap the classes of the labels.
        For each type of noise, the configuration can be set using the `noise_config` parameter. Refer to NoisyCocoDetection class for more details.
        Also, it supports k-fold cross validation by splitting the dataset into `num_folds` folds and using the `fold_index` to select the fold.

        :param root: The root directory of the dataset.
        :param annFile: The annotation file of the dataset.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        :param train_transform: The transform to apply to the train images. Defaults to `None`.
        :param valid_transform: The transform to apply to the valid images. Defaults to `None`.
        :param test_transform: The transform to apply to the test images. Defaults to `None`.
        :param task: The task to perform. Either `"det"` for detection or `"seg"` for segmentation. Defaults to `"det"`.
        :param noise_type: The type of noise to apply. Either `"overlook"`, `"badloc"`, or `"swap"`. Defaults to `"overlook"`.
        :param noise_config: The configuration of the noise to apply. Defaults to `OverlookNoiseConfig()`.
        :param fold_index: The index of the fold to use. Defaults to `0`.
        :param num_folds: The number of folds to use. Defaults to `4`.
        """
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train = None
        self.data_val = None
        self.data_pred = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val:  
            self.dataset = NoisyCocoDetection(
                type=self.hparams.noise_type,
                config=self.hparams.noise_config,
                task=self.hparams.task,
                root = self.hparams.root,
                annFile = self.hparams.annFile, 
                transform=None
            )
            self.num_classes = len(self.dataset.classes)

            # the seed is intentionally fixed to ensure the same split for each instance, which is crucial for our implementation
            kfold = KFold(n_splits=self.hparams.num_folds, shuffle=True, random_state=42)
            splitted_data = [k for k in kfold.split(list(range(len(self.dataset))))]

            train_idxs, val_idxs = splitted_data[self.hparams.fold_index]

            self.data_train = SubsetWithTransform(self.dataset, train_idxs, self.hparams.train_transform)
            self.data_val = SubsetWithTransform(self.dataset, val_idxs, self.hparams.valid_transform)
            self.data_pred = SubsetWithTransform(self.dataset, val_idxs, self.hparams.test_transform)

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=collate_fn
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_pred,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn
        )
    
    def pred_images(self):
        # remove the transform to get the original images without any augmentation e.g. normalization
        transform = self.data_pred.transform
        self.data_pred.transform = None 
        images = [image for image, _ in self.data_pred]
        self.data_pred.transform = transform
        return images
    
    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass