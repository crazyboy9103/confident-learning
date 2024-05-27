import torch
from torch import optim
from torchmetrics.detection.mean_ap import MeanAveragePrecision as mAP
import lightning.pytorch as pl

from .builder import retinanet_builder

class DetectionModel(pl.LightningModule):
    def __init__(
        self, 
        num_classes: int,
        trainable_backbone_layers: int,
        optimizer: optim.Optimizer, 
        scheduler: optim.lr_scheduler, 
        fold: int,
        compile: bool = False, 
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.net = retinanet_builder(num_classes, trainable_backbone_layers, **kwargs)

        self.mAP = mAP(
            box_format='xyxy',
            iou_type = 'bbox', 
            iou_thresholds = [0.5,],
            extended_summary = False, 
            backend = "faster_coco_eval"
        )

    def forward(self, images, targets=None):
        return self.net(images, targets)

    def shared_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images, targets)
        return outputs
    
    def training_step(self, batch, batch_idx):
        loss_dict = self.shared_step(batch, batch_idx)
        self.log_dict({f"train_loss_{self.hparams.fold}_{k}": v for k, v in loss_dict.items()})
        return sum(loss_dict.values())
    
    def validation_step(self, batch, batch_idx):
        preds = self.shared_step(batch, batch_idx)
        targets = batch[1]
        self.mAP(preds, targets)

    def on_validation_epoch_end(self):
        result = self.mAP.compute()
        result.pop("classes")
        self.log_dict({f"valid_{k}_{self.hparams.fold}": v for k, v in result.items()}) 
        self.mAP.reset()

    def predict_step(self, batch, batch_idx):
        _, targets = batch
        preds = self.shared_step(batch, batch_idx)
        return targets, preds
    
    def setup(self, stage: str):
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
    
    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(
            self.parameters()
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