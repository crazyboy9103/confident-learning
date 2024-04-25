import torch
from torch import nn
from torch import optim
from torchmetrics.classification import JaccardIndex, Accuracy
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
        aux_loss: bool = True
    ):
        super().__init__()
        self.save_hyperparameters()

        self.net = fcn_builder(num_classes, aux_loss)

        self.train_jac = JaccardIndex(
            task="multiclass",
            num_classes=num_classes,
            average="macro",
            ignore_index=255
        )
        self.valid_jac = JaccardIndex(
            task="multiclass",
            num_classes=num_classes,
            average="macro",
            ignore_index=255
        )
    
        self.train_acc = Accuracy(
            task="multiclass",
            num_classes=num_classes,
            average="macro",
            ignore_index=255
        )
        self.valid_acc = Accuracy(
            task="multiclass",
            num_classes=num_classes,
            average="macro",
            ignore_index=255
        )

    def criterion(self, inputs, targets):
        losses = {}
        for name, x in inputs.items():
            losses[name] = nn.functional.cross_entropy(x, targets, ignore_index=255)
        
        if len(losses) == 1:
            return losses["out"]
        
        return losses["out"] + 0.5 * losses["aux"]
    
    def forward(self, images):
        return self.net(images)
    
    def shared_step(self, batch, batch_idx):
        images, targets = batch
        images = torch.stack(images)
        target_masks = torch.cat([target["masks"] for target in targets])
        outputs = self(images)
        loss = self.criterion(outputs, target_masks)
        return loss, outputs, target_masks
    
    def training_step(self, batch, batch_idx):
        loss, preds, target_masks = self.shared_step(batch, batch_idx)

        pred_mask = preds["out"].argmax(dim=1)
        self.train_jac(pred_mask, target_masks)
        self.train_acc(pred_mask, target_masks)

        self.log(f"train_loss_{self.hparams.fold}", loss)
        return loss
    
    def on_train_epoch_end(self):
        self.log(f"train_jaccard_{self.hparams.fold}", self.train_jac.compute())
        self.log(f"train_acc_{self.hparams.fold}", self.train_acc.compute())
        self.train_jac.reset()
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        loss, preds, target_masks = self.shared_step(batch, batch_idx)

        pred_mask = preds["out"].argmax(dim=1)
        self.valid_jac(pred_mask, target_masks)
        self.valid_acc(pred_mask, target_masks)

        self.log(f"valid_loss_{self.hparams.fold}", loss)
    
    def on_validation_epoch_end(self):
        self.log(f"valid_jaccard_{self.hparams.fold}", self.valid_jac.compute())
        self.log(f"valid_acc_{self.hparams.fold}", self.valid_acc.compute())
        self.valid_jac.reset()
        self.valid_acc.reset()

    def predict_step(self, batch, batch_idx):
        _, targets = batch
        _, preds, _ = self.shared_step(batch, batch_idx)
        preds = preds["out"].softmax(dim=1)
        return targets, preds 
        
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