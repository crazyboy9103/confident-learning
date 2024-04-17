import torch
from torch import nn
from torch import optim
from torchmetrics.detection.mean_ap import MeanAveragePrecision as mAP
import lightning.pytorch as pl

from .builder import fcn_builder

class SegmentationModel(pl.LightningModule):
    def __init__(
        self, 
        num_classes: int, 
        optimizer: optim.Optimizer, 
        scheduler: optim.lr_scheduler, 
        fold: int,
        compile: bool = False, 
    ):
        super().__init__()
        self.save_hyperparameters()

        self.net = fcn_builder(num_classes)

        self.mAP = mAP(
            box_format='xyxy',
            iou_type = 'segm', 
            iou_thresholds = [0.5,],
            extended_summary = False, 
            backend = "faster_coco_eval"
        )

    def forward(self, x):
        return self.net(x)

    def criterion(self, inputs, targets):
        losses = {}
        for name, x in inputs.items():
            losses[name] = nn.functional.cross_entropy(x, targets)
        return losses
    
    def shared_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        return targets, outputs
    
    def training_step(self, batch, batch_idx):
        targets, outputs = self.shared_step(batch, batch_idx)
        loss_dict = self.criterion(outputs, targets)
        return loss_dict
    
    def validation_step(self, batch, batch_idx):
        targets, outputs = self.shared_step(batch, batch_idx)
        self.mAP(outputs, targets)

    def on_validation_epoch_end(self):
        self.log_dict({f"{k}_{self.hparams.fold}": v for k, v in self.mAP.compute().items()}) 
        self.mAP.reset()

    def predict_step(self, batch, batch_idx):
        targets, outputs = self.shared_step(batch, batch_idx)
        return batch[0], targets, outputs
    
    def setup(self, stage: str):
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
    
    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(
            self.trainer.model.parameters()
        )
        if self.hparams.scheduler:
            scheduler = self.hparams.scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                }
            }

        return {
            "optimizer": optimizer
        }