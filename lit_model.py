import lightning.pytorch as pl
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy
import timm

class LitModel(pl.LightningModule):
    def __init__(self, model, model_params, fold, image_key, label_key, optimizer, optimizer_params, scheduler, scheduler_params):
        super().__init__()
        # To obtain embeddings and logits from the model
        self.backbone = timm.create_model(model, pretrained=model_params["pretrained"], num_classes=0)
        self.classifier = nn.LazyLinear(model_params["num_classes"], bias = True)
        
        self.train_acc = Accuracy(task="multiclass", num_classes=model_params["num_classes"])
        self.valid_acc = Accuracy(task="multiclass", num_classes=model_params["num_classes"])

        self.save_hyperparameters()

    def forward(self, image):
        embeddings = self.backbone(image)
        logits = self.classifier(embeddings)
        return embeddings, logits

    def shared_step(self, batch, batch_idx):
        x, y = batch[self.hparams.image_key], batch[self.hparams.label_key]
        embeddings, logits = self(x)
        loss = F.cross_entropy(logits, y)
        probs = F.softmax(logits, dim=1)
        return loss, embeddings, probs
    
    def training_step(self, batch, batch_idx):
        loss, _, probs = self.shared_step(batch, batch_idx)
        preds = torch.argmax(probs, dim=1)
        self.train_acc(preds, batch[self.hparams.label_key])
        self.log(f'train_loss_folditer{self.hparams.fold}', loss)
        return loss 
    
    def on_train_epoch_end(self):
        self.log(f'train_acc_epoch_folditer{self.hparams.fold}', self.train_acc.compute())
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        loss, _, probs = self.shared_step(batch, batch_idx)
        preds = torch.argmax(probs, dim=1)
        self.valid_acc(preds, batch[self.hparams.label_key])
        self.log(f'valid_loss_folditer{self.hparams.fold}', loss)

    def on_validation_epoch_end(self):
        self.log(f'valid_acc_epoch_folditer{self.hparams.fold}', self.valid_acc.compute())
        self.valid_acc.reset()

    def predict_step(self, batch, batch_idx):
        _, embeddings, probs = self.shared_step(batch, batch_idx)
        return embeddings, probs
    
    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(self.parameters(), **self.hparams.optimizer_params)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": self.hparams.scheduler(optimizer, **self.hparams.scheduler_params),
                "interval": "step",
                "frequency": 1,
            }
        }
    
if __name__ == "__main__":
    random_input = torch.rand(1, 3, 224, 224)
    from scheduler import LinearWarmUpMultiStepDecay
    model = LitModel("resnet18", {"num_classes": 10, "pretrained": True}, 1, torch.optim.Adam, {"lr": 1e-3}, LinearWarmUpMultiStepDecay, {"initial_lr": 1e-3, "milestones": [10, 20], "warmup_steps": 5})
    print(model.hparams)
    print(model(random_input))    
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
    model = LitModel("resnet50", {"num_classes": 10, "pretrained": False}, 1, torch.optim.SGD, {"lr": 1e-3}, CosineAnnealingWarmRestarts, {"T_0": 30, "T_mult": 1, "eta_min": 1e-3, "last_epoch": -1, "verbose": False})
    print(model.hparams)
    print(model(random_input))
    model = LitModel("efficientnet_b0", {"num_classes": 10, "pretrained": True}, 1, torch.optim.Adadelta, {"lr": 1e-3}, ReduceLROnPlateau, {"mode": "min", "factor": 0.1, "patience": 10, "threshold": 0.0001, "threshold_mode": "rel", "cooldown": 0, "min_lr": 0, "eps": 1e-08, "verbose": False})
    print(model.hparams)
    print(model(random_input))
