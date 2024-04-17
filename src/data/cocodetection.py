from typing import Any, Dict, List, Optional, Callable, Literal, Union
import random
from itertools import combinations
from dataclasses import dataclass

import numpy as np
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from sklearn.model_selection import KFold
import cv2

from .utils import SubsetWithTransform

def polygon_area(polygon):
    reshaped_polygon = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    return cv2.contourArea(reshaped_polygon)

def collate_fn(batch):
    return tuple(zip(*batch))

@dataclass
class OverlookNoiseConfig:
    prob: float = 0.05
    debug: bool = False # if True, do not remove the noisy labels

@dataclass
class BadLocNoiseConfig:
    prob: float = 0.05
    max_pixel: int = 50

@dataclass
class SwapNoiseConfig:
    num_classes_to_swap: int = 3
    prob: float = 0.05

class NoisyCocoDetection(CocoDetection):
    def _load_target(self, id: int):
        target = super()._load_target(id)

        if self.type == "overlook" and not self.config.debug:
            target = [ann for ann in target if not ann["noisy_label"]]
            
        return target

    def __init__(self, type: Literal["overlook", "badloc", "swap"], config: Union[OverlookNoiseConfig, BadLocNoiseConfig, SwapNoiseConfig], *args, **kwargs):
        super().__init__(*args, **kwargs)
        random.seed(42) # fixed seed for reproducibility

        # CocoDetection uses self.coco.loadAnns(self.coco.getAnnIds(id)) to load the target, where 
        #   imgToAnns: imgToAnns['image_id'] = anns
        #   getAnnIds: self.imgToAnns[imgId]
        #   loadAnns: [self.anns[id] for id in annIds]
        # Therefore, manipulate self.coco.imgToAnns for overlook by removing some anns from imgToAnns
        # This requires the following steps:
        # 1. For given prob, calculate the number of annotations to remove
        # 2. Randomly sample the annotations to remove
        # 3. Remove the sampled annotations from imgToAnns

        # Then, manipulate self.coco.anns for badloc/swap by perturbing the positions/classes
        # This requires the following steps:
        # 1. For given prob, calculate the number of annotations to perturb
        #   1.1. For swap, randomly sample the classes to swap for given num_classes_to_swap
        #   1.2. Create pairs of classes to swap
        #   1.3. For each pair, randomly sample the annotations to swap for given prob
        #   1.4. Swap the classes of the sampled annotations
        # 2. Randomly sample the annotations to perturb
        # 3. Perturb the positions/classes of the sampled annotations

        self.type = type
        self.config = config

        # Add a flag to each annotation to track whether it is noisy or not
        for ann_id, ann in self.coco.anns.items():
            ann["noisy_label"] = False

        cat_ids = self.coco.getCatIds()
        self.classes = cat_ids

        ann_ids = self.coco.getAnnIds()
        
        counts = {}
        for cat_id in cat_ids:
            counts[cat_id] = len(self.coco.getAnnIds(catIds=cat_id))

        match type:
            case "overlook":
                assert isinstance(config, OverlookNoiseConfig)
                prob = config.prob

                num_noise = int(len(ann_ids) * prob)
                ann_ids_overlook = random.sample(ann_ids, num_noise)

                for ann_id in ann_ids_overlook:
                    self.coco.anns[ann_id]["noisy_label"] = True

            case "badloc":
                assert isinstance(config, BadLocNoiseConfig)
                prob = config.prob
                max_pixel = config.max_pixel

                num_noise = int(len(ann_ids) * prob)
                ann_ids_badloc = random.sample(ann_ids, num_noise)

                for ann_id in ann_ids_badloc:
                    self.coco.anns[ann_id]["noisy_label"] = True

                    ann = self.coco.anns[ann_id]
                    
                    x1, y1, w, h = ann["bbox"]
                    x2, y2 = x1 + w, y1 + h

                    segmentation = ann["segmentation"][0] # zeroth element is the segmentation

                    # Fetch width and height for clipping within the image
                    img = self.coco.imgs[ann["image_id"]]
                    img_width, img_height = img["width"], img["height"]

                    # Pixels to move the bounding box 
                    dx = random.randint(-max_pixel, max_pixel)
                    dy = random.randint(-max_pixel, max_pixel)

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

                    ann["area"] = w * h
                    ann["bbox"] = [x1, y1, w, h]

                    # Move the segmentation
                    segmentation = np.array(segmentation).reshape(-1, 2)
                    segmentation[:, 0] = np.clip(segmentation[:, 0] + dx, 0, img_width-1)
                    segmentation[:, 1] = np.clip(segmentation[:, 1] + dy, 0, img_height-1)
                    ann["segmentation"] = [segmentation.flatten().tolist()]
                    ann["area"] = polygon_area(ann["segmentation"])

            case "swap":
                assert isinstance(config, SwapNoiseConfig)
                num_classes_to_swap = config.num_classes_to_swap
                prob = config.prob

                classes_to_swap = random.sample(cat_ids, num_classes_to_swap)
                pairs_to_swap = combinations(classes_to_swap, 2)
                
                ann_ids_class = {cat_id: self.coco.getAnnIds(catIds=cat_id) for cat_id in classes_to_swap}
                counts = {cat_id: len(ann_ids) for cat_id, ann_ids in ann_ids_class.items()}

                for i, j in pairs_to_swap:
                    ann_ids_i = ann_ids_class[i]
                    ann_ids_j = ann_ids_class[j]

                    num_to_swap_i = int(min(counts[i], len(ann_ids_class[i])) * prob)
                    num_to_swap_j = int(min(counts[j], len(ann_ids_class[j])) * prob)

                    ann_ids_swap_i = random.sample(ann_ids_i, num_to_swap_i)
                    ann_ids_swap_j = random.sample(ann_ids_j, num_to_swap_j)
                    
                    for ann_id in ann_ids_swap_i:
                        self.coco.anns[ann_id]["noisy_label"] = True

                        ann = self.coco.anns[ann_id]
                        ann["category_id"] = j
                    
                    for ann_id in ann_ids_swap_j:
                        self.coco.anns[ann_id]["noisy_label"] = True

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
        noise_type: Literal["overlook", "badloc", "swap"] = "overlook",
        noise_config: Union[OverlookNoiseConfig, BadLocNoiseConfig, SwapNoiseConfig] = OverlookNoiseConfig(),
        fold_index: int = 0, # for k-fold cross validation
        num_folds: int = 4, # for k-fold cross validation
    ) -> None:
        """Initialize a `NoisyCocoDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        :param train_transform: The transform to apply to the train images. Defaults to `None`.
        :param valid_transform: The transform to apply to the valid images. Defaults to `None`.
        :param test_transform: The transform to apply to the test images. Defaults to `None`.
        :param noise_type: The type of noise to apply. Defaults to `"overlook"`.
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
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_pred,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn
        )
    
    def predict_info(self):
        return {
            "flipped_labels": [self.dataset.flipped_labels()[i] for i in self.data_pred.idxs],
            "pred_idxs": self.data_pred.idxs,
            "fold_index": self.hparams.fold_index
        }
    
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