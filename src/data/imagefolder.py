from typing import Any, Dict, Optional, Callable
import random
from itertools import combinations

import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.model_selection import StratifiedKFold

from .utils import SubsetWithTransform

class NoisyImageFolder(ImageFolder):
    def __init__(self, num_classes_to_swap = 3, swap_prob = 0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        random.seed(42) # fixed seed for reproducibility

        self.noisy_labels = [t for t in self.targets]
        self.swap_labels(num_classes_to_swap, swap_prob)

    def swap_labels(self, num_classes_to_swap, swap_prob):
        """Flips the targets with a given probability.
        Args:
            num_classes_to_swap (int): The number of classes to swap <= num_classes.
            swap_prob (float): The probability to flip the targets.
        """
        class_idxs = {}
        for i, target in enumerate(self.targets):
            class_idxs.setdefault(target, set()).add(i)
        
        counts = {k: len(v) for k, v in class_idxs.items()}
        # deprecation random.sample from a set, so convert to tuple
        classes_to_swap = random.sample(tuple(counts.keys()), num_classes_to_swap)
        pairs_to_swap = combinations(classes_to_swap, 2)

        for i, j in pairs_to_swap:
            # In case the number of samples to swap is less than the number of samples left in the class
            num_samples_to_swap_i = int(min(counts[i], len(class_idxs[i])) * swap_prob)
            num_samples_to_swap_j = int(min(counts[j], len(class_idxs[j])) * swap_prob)

            idxs_i = set(random.sample(tuple(class_idxs[i]), num_samples_to_swap_i))
            idxs_j = set(random.sample(tuple(class_idxs[j]), num_samples_to_swap_j))

            # swap the labels
            for idx in idxs_i:
                self.noisy_labels[idx] = j

            for idx in idxs_j:
                self.noisy_labels[idx] = i
            
            # update the class indices
            class_idxs[i] = class_idxs[i] - idxs_i
            class_idxs[j] = class_idxs[j] - idxs_j

    def __getitem__(self, index: int):
        img, _ = super().__getitem__(index)
        return img, self.noisy_labels[index]
    

class NoisyImageFolderDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_transform: Callable = None, # for image augmentation Callable[PIL.Image]
        valid_transform: Callable = None, # for image augmentation Callable[PIL.Image]
        test_transform: Callable = None, # for image augmentation Callable[PIL.Image]
        num_classes_to_swap: int = 3, # for noisy labels
        swap_prob: float = 0.1, # for noisy labels
        fold_index: int = 0, # for k-fold cross validation
        num_folds: int = 4, # for k-fold cross validation
    ) -> None:
        """Initialize a `ImageFolderDataModule`.

        :param root: The data directory. Defaults to `"data/"`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        :param train_transform: The transform to apply to the train images. Defaults to `None`.
        :param valid_transform: The transform to apply to the valid images. Defaults to `None`.
        :param test_transform: The transform to apply to the test images. Defaults to `None`.
        :param num_classes_to_swap: The number of classes to swap <= num_classes. Defaults to `3`.
        :param swap_prob: The probability to flip the targets. Defaults to `0.1`.
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
            self.dataset = NoisyImageFolder(
                root = self.hparams.root, 
                num_classes_to_swap=self.hparams.num_classes_to_swap,
                swap_prob=self.hparams.swap_prob, 
                transform=None
            )
            self.num_classes = len(self.dataset.classes)

            # the seed is intentionally fixed to ensure the same split for each instance, which is crucial for our implementation
            kfold = StratifiedKFold(n_splits=self.hparams.num_folds, shuffle=True, random_state=42)
            splitted_data = [k for k in kfold.split(list(range(len(self.dataset))), self.dataset.noisy_labels)]
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
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_pred,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    
    def predict_info(self):
        return {
            "orig_labels": [self.dataset.targets[i] for i in self.data_pred.idxs],
            "noisy_labels": [self.dataset.noisy_labels[i] for i in self.data_pred.idxs],
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