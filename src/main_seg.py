import sys
sys.path.append("..")

import torch
import lightning.pytorch as pl 
import hydra 
from omegaconf import DictConfig
import numpy as np
import pandas as pd
from PIL import Image
import wandb 

from conflearn.classification import Classification
from utils import classification_evaluation

@hydra.main(version_base=None, config_path="../configs", config_name="test_cla.yaml")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    data: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer)
    logger = hydra.utils.instantiate(cfg.logger)

    num_folds: int = cfg.get("num_folds", 4)
    
    accum_images = []
    accum_targets = []
    accum_probs = []
    accum_swapped_labels = []

    for k in range(num_folds):
        data_module = data(fold_index=k)
        data_module.setup()

        model_instance = model(fold=k, num_classes=data_module.num_classes)

        callbacks = []
        for callback_name, callback_cfg in cfg.callbacks.items():
            callback: pl.Callback = hydra.utils.instantiate(callback_cfg)

            if callback_name == "model_checkpoint":
                callback = callback(filename=f"cla_model_{cfg.task_name}_{k}", monitor=f"valid_acc_{k}", mode="max")

            callbacks.append(callback)

        trainer_instance = trainer(logger=logger, callbacks=callbacks)
        trainer_instance.fit(model_instance, datamodule=data_module)
        preds = trainer_instance.predict(model_instance, datamodule=data_module)
        
        targets = torch.cat([result[0] for result in preds], dim=0)
        probs = torch.cat([result[1] for result in preds], dim=0)

        accum_targets.append(targets)
        accum_probs.append(probs)

        swapped_labels = data_module.pred_swapped_labels()
        # extend swapped_labels as they are not batched
        accum_swapped_labels.extend(swapped_labels)

        images = data_module.pred_images()
        accum_images.extend(images)
    
    targets = torch.cat(accum_targets, dim=0).numpy()
    probs = torch.cat(accum_probs, dim=0).numpy()
    preds = np.argmax(probs, axis=1)
    swapped_labels = np.array(accum_swapped_labels)
    images = [image.transpose(1, 2, 0) for image in images] # (C, H, W) -> (H, W, C)
    pil_images = [Image.fromarray(image.astype(np.uint8)) for image in images]

    decoded_targets = data_module.idx_to_class(targets)
    decoded_preds = data_module.idx_to_class(preds)

    # log the classification evaluation
    # conf_mat_fig, cls_report = classification_evaluation(targets, preds)
    # logger.experiment.log({"cls_confusion_matrix": wandb.Image(conf_mat_fig)})
    # logger.experiment.log({
    #     'cls_accuracy': cls_report['accuracy'],
    #     'cls_macro_avg_precision': cls_report['macro avg']['precision'],
    #     'cls_macro_avg_recall': cls_report['macro avg']['recall'],
    #     'cls_macro_avg_f1': cls_report['macro avg']['f1-score'],
    #     'cls_weighted_avg_precision': cls_report['weighted avg']['precision'],
    #     'cls_weighted_avg_recall': cls_report['weighted avg']['recall'],
    #     'cls_weighted_avg_f1': cls_report['weighted avg']['f1-score']
    # })

    conflearn = Classification(targets, probs, data_module.num_classes)
    error_mask, scores = conflearn.get_result(cfg.cl_method, cfg.cl_score_method)
    
    # log the evaluation metrics for finding the label issues
    conf_mat_fig, cls_report = classification_evaluation(swapped_labels, error_mask)
    logger.experiment.log({"conflearn_confusion_matrix": wandb.Image(conf_mat_fig)})
    logger.experiment.log({
        'conflearn_accuracy': cls_report['accuracy'],
        'conflearn_macro_avg_precision': cls_report['macro avg']['precision'],
        'conflearn_macro_avg_recall': cls_report['macro avg']['recall'],
        'conflearn_macro_avg_f1': cls_report['macro avg']['f1-score'],
        'conflearn_weighted_avg_precision': cls_report['weighted avg']['precision'],
        'conflearn_weighted_avg_recall': cls_report['weighted avg']['recall'],
        'conflearn_weighted_avg_f1': cls_report['weighted avg']['f1-score']
    })

    df = pd.DataFrame({
        "given_label": decoded_targets,
        "pred_label": decoded_preds,
        "swapped": swapped_labels,
        "label_issue": error_mask,
        "label_score": scores,
        "images": [wandb.Image(image) for image in pil_images]
    })
    logger.experiment.log({"result": wandb.Table(dataframe=df)})

if __name__ == '__main__':
    main()