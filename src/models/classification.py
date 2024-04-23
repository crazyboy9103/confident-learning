import torch
from torch.nn import functional as F
from torch import optim
from torchmetrics import Accuracy
import lightning.pytorch as pl

from .builder import efficientnetb0_builder

class ClassificationModel(pl.LightningModule):
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
        
        self.net = efficientnetb0_builder(num_classes)

        self.train_acc = Accuracy(task = "multiclass", num_classes=num_classes)
        self.valid_acc = Accuracy(task = "multiclass", num_classes=num_classes)


    def forward(self, x):
        return self.net(x)

    def shared_step(self, batch, batch_idx):
        images, targets = batch
        logits = self(images)
        loss = F.cross_entropy(logits, targets)
        probs = F.softmax(logits, dim=1)
        return loss, probs
    
    def training_step(self, batch, batch_idx):
        loss, probs = self.shared_step(batch, batch_idx)
        preds = torch.argmax(probs, dim=1)
        self.train_acc(preds, batch[1])
        self.log(f"train_loss_{self.hparams.fold}", loss)
        return loss
    
    def on_train_epoch_end(self):
        self.log(f"train_acc_{self.hparams.fold}", self.train_acc.compute())
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        loss, probs = self.shared_step(batch, batch_idx)
        preds = torch.argmax(probs, dim=1)
        self.valid_acc(preds, batch[1])
        self.log(f"valid_loss_{self.hparams.fold}", loss)
        return loss

    def on_validation_epoch_end(self):
        self.log(f'valid_acc_{self.hparams.fold}', self.valid_acc.compute())
        self.valid_acc.reset()
    
    def on_train_epoch_end(self):
        self.log(f"train_acc_{self.hparams.fold}", self.train_acc.compute())
        self.train_acc.reset()

    def predict_step(self, batch, batch_idx):
        _, targets = batch
        _, probs = self.shared_step(batch, batch_idx)
        return targets, probs
    
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