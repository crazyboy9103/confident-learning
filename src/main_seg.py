import sys
sys.path.append("..")
from itertools import chain

import torch
import lightning.pytorch as pl 
import hydra 
from omegaconf import DictConfig
import pandas as pd
import wandb 

from conflearn.segmentation import Segmentation
from src.eval_utils import roc_evaluation

def prepare_for_conflearn(target_dict, pred_prob):
    self_conf = torch.gather(pred_prob, dim=0, index=target_dict["masks"].long())
    self_conf = self_conf.squeeze(0)
    return self_conf.numpy()

def parse_masks_for_wandb(target_mask_per_image, pred_mask_per_image, class_labels):
    # explicitly cast to int and float to avoid serialization issues
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
    
    accum_targets = []
    accum_preds = []
    accum_images = []

    for k in range(num_folds):
        data_module = data(fold_index=k)
        data_module.setup()

        model_instance = model(fold=k, num_classes=data_module.num_classes + 1)

        callbacks = []
        for callback_name, callback_cfg in cfg.callbacks.items():
            callback: pl.Callback = hydra.utils.instantiate(callback_cfg)

            if callback_name == "model_checkpoint":
                callback = callback(filename=f"seg_model_{cfg.task_name}_{k}", monitor=f"valid_acc_{k}", mode="max")

            callbacks.append(callback)

        trainer_instance = trainer(logger=logger, callbacks=callbacks)
        trainer_instance.fit(model_instance, datamodule=data_module)
        preds = trainer_instance.predict(model_instance, datamodule=data_module)
        
        targets = [result[0] for result in preds]
        preds = [result[1] for result in preds]

        targets = list(chain.from_iterable(targets))
        preds = list(chain.from_iterable(preds))

        accum_targets.extend(targets)
        accum_preds.extend(preds)          
        
        images = data_module.pred_images()
        accum_images.extend(images)

    # (W x H), (1 x H x W), (K+1 x H x W)
    print(accum_images[0].size, accum_targets[0]["masks"].shape, accum_preds[0].shape)

    self_confs = []
    for target, pred in zip(accum_targets, accum_preds):
        self_confs.append(prepare_for_conflearn(target, pred))

    conflearn = Segmentation(self_confs)
    scores = conflearn.get_result(cfg.pooling, cfg.softmin_temperature)

    # Consider the image is noisy if any of the annotations for that image is noisy 
    # otherwise, the image is normal 
    noisy_labels = [any(target["noisy_labels"]) for target in accum_targets]
    class_map = data_module.dataset.classes # {1: {"id": 1, "name": "person"}, 2: {"id": 2, "name": "car"}}

    class_labels = {class_id: cat["name"] for class_id, cat in class_map.items()}

    class_set = wandb.Classes([
        {"name": name, "id": class_id} for class_id, name in class_labels.items()
    ])

    df = pd.DataFrame({
        "noisy": noisy_labels,
        "label_score": scores,
        "images": [wandb.Image(
            accum_images[i],
            masks = parse_masks_for_wandb(accum_targets[i]["masks"].squeeze(0).numpy(), accum_preds[i].argmax(0).numpy(), class_labels),
            classes=class_set
        ) for i in range(len(accum_images))]
    })
    logger.experiment.log({"result": wandb.Table(dataframe=df)})

    aucroc, best_threshold, roc_curve_fig = roc_evaluation(noisy_labels, scores)

    logger.experiment.log({
        'aucroc': aucroc,
        'best_threshold': best_threshold,
        'roc_curve': wandb.Image(roc_curve_fig)
    })
    

if __name__ == '__main__':
    main()