import sys
sys.path.insert(0, "/workspace/confident-learning")

from itertools import chain

import torch
import lightning.pytorch as pl 
import hydra 
from omegaconf import DictConfig
import pandas as pd
import wandb 

from conflearn.segmentation import Segmentation
from eval_utils import roc_evaluation

def configure_callbacks(cfg, fold_index):
    callbacks = []
    for callback_name, callback_cfg in cfg.callbacks.items():
        callback: pl.Callback = hydra.utils.instantiate(callback_cfg)

        if callback_name == "model_checkpoint":
            callback = callback(filename=f"seg_model_{cfg.task_name}_{fold_index}", monitor=f"valid_acc_{fold_index}", mode="max")

        callbacks.append(callback)
    return callbacks

def train_and_predict(model, data_module, trainer, logger, callbacks, accum_data):
    trainer_instance = trainer(logger=logger, callbacks=callbacks)
    trainer_instance.fit(model, datamodule=data_module)
    preds = trainer_instance.predict(model, datamodule=data_module, ckpt_path="best")
    
    targets = [result[0] for result in preds]
    preds = [result[1] for result in preds]

    targets = list(chain.from_iterable(targets))
    preds = list(chain.from_iterable(preds))

    accum_data["targets"].extend(targets)
    accum_data["preds"].extend(preds)          
    accum_data["images"].extend(data_module.pred_images())

def process_accumulated_data(accum_data):
    self_confs = []
    for target, pred in zip(accum_data["targets"], accum_data["preds"]):
        self_conf = torch.gather(pred, dim=0, index=target["masks"].long())
        self_conf = self_conf.squeeze(0).numpy()
        self_confs.append(self_conf)
    return self_confs

def evaluate_and_log(cfg, accum_data, self_confs, data_module, logger):
    conflearn = Segmentation(self_confs)
    scores = conflearn.get_result(cfg.pooling, cfg.softmin_temperature)

    # Consider the image is noisy if any of the annotations for that image is noisy 
    # otherwise, the image is normal 
    noisy_labels = [any(target["noisy_labels"]) for target in accum_data["targets"]]
    class_map = data_module.dataset.classes # {1: {"id": 1, "name": "person"}, 2: {"id": 2, "name": "car"}}

    class_labels = {class_id: cat["name"] for class_id, cat in class_map.items()}

    class_set = wandb.Classes([
        {"name": name, "id": class_id} for class_id, name in class_labels.items()
    ])

    df = pd.DataFrame({
        "swapped": noisy_labels,
        "label_score": scores,
        "images": [wandb.Image(
            accum_data["images"][i],
            masks = parse_masks_for_wandb(
                accum_data["targets"][i]["masks"].squeeze(0).numpy(), 
                accum_data["preds"][i].argmax(0).numpy(), 
                class_labels
            ),
            classes=class_set
        ) for i in range(len(accum_data["images"]))]
    })
    logger.experiment.log({"result": wandb.Table(dataframe=df)})

    aucroc, best_threshold, roc_curve_fig = roc_evaluation(noisy_labels, scores)
    logger.experiment.log({
        'aucroc': aucroc,
        'best_threshold': best_threshold,
        'roc_curve': wandb.Image(roc_curve_fig)
    })

def parse_masks_for_wandb(target_mask_per_image, pred_mask_per_image, class_labels):
    predictions = {
        "mask_data": pred_mask_per_image,
        "class_labels": class_labels
    }

    ground_truths = {
        "mask_data": target_mask_per_image,
        "class_labels": class_labels
    }
    return {
        "predictions": predictions,
        "ground_truths": ground_truths
    }
    
@hydra.main(version_base=None, config_path="../configs", config_name="test_seg.yaml")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    data: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer)
    logger = hydra.utils.instantiate(cfg.logger)

    num_folds: int = cfg.get("num_folds", 4)
    
    accum_data = {'images': [], 'targets': [], 'preds': []}

    for k in range(num_folds):
        data_module = data(fold_index=k)
        data_module.setup()

        model_instance = model(fold=k, num_classes=data_module.num_classes + 1)
        callbacks = configure_callbacks(cfg, k)
        
        train_and_predict(model_instance, data_module, trainer, logger, callbacks, accum_data)
        
    self_confs = process_accumulated_data(accum_data)
    evaluate_and_log(cfg, accum_data, self_confs, data_module, logger)

    
if __name__ == '__main__':
    main()