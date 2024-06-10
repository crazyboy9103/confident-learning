import sys
sys.path.insert(0, "/workspace/confident-learning")

import torch
import lightning.pytorch as pl 
import hydra 
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import wandb 
from cleanlab.filter import find_label_issues, get_label_quality_scores

from conflearn.classification import Classification
from eval_utils import classification_evaluation

def configure_callbacks(cfg, fold_index):
    callbacks = []
    for callback_name, callback_cfg in cfg.callbacks.items():
        callback: pl.Callback = hydra.utils.instantiate(callback_cfg)

        if callback_name == "model_checkpoint":
            callback = callback(filename=f"cla_model_{cfg.task_name}_{fold_index}", monitor=f"valid_acc_{fold_index}", mode="max")

        callbacks.append(callback)
    return callbacks

def train_and_predict(model, data_module, trainer, logger, callbacks, accum_data):
    trainer_instance = trainer(logger=logger, callbacks=callbacks)
    trainer_instance.fit(model, datamodule=data_module)
    preds = trainer_instance.predict(model, datamodule=data_module, ckpt_path="best")

    # Collect and accumulate predictions

    targets = torch.cat([result[0] for result in preds], dim=0)
    probs = torch.cat([result[1] for result in preds], dim=0)
    
    accum_data['targets'].append(targets)
    accum_data['probs'].append(probs)
    accum_data['swapped_labels'].extend(data_module.pred_swapped_labels())
    accum_data['images'].extend(data_module.pred_images())


def process_accumulated_data(accum_data):
    targets = torch.cat(accum_data['targets'], dim=0).numpy()
    probs = torch.cat(accum_data['probs'], dim=0).numpy()
    preds = np.argmax(probs, axis=1)
    swapped_labels = np.array(accum_data['swapped_labels'])
    return targets, probs, preds, swapped_labels

def evaluate_and_log(cfg, images, targets, probs, preds, swapped_labels, data_module, logger):
    decoded_targets = [data_module.idx_to_class(idx) for idx in targets]
    decoded_preds = [data_module.idx_to_class(idx) for idx in preds]

    print("Computing the label issues...")
    conflearn = Classification(targets, probs, data_module.num_classes)
    error_mask, scores = conflearn.get_result(cfg.cl_method, cfg.cl_score_method)
    
    cleanlab_indices = find_label_issues(targets, probs)
    cleanlab_error_mask = np.zeros_like(targets, dtype=bool)
    cleanlab_error_mask[cleanlab_indices] = True
    cleanlab_scores = get_label_quality_scores(targets, probs)
    print("Evaluating the label issues...")
    # log the evaluation metrics for finding the label issues
    conf_mat_fig, cls_report = classification_evaluation(swapped_labels, error_mask)

    logger.experiment.log({"conflearn_confusion_matrix": wandb.Image(conf_mat_fig)})
    logger.experiment.log({
        'conflearn_accuracy': cls_report['accuracy'],
        'conflearn_precision': cls_report['macro avg']['precision'],
        'conflearn_recall': cls_report['macro avg']['recall'],
        'conflearn_f1': cls_report['macro avg']['f1-score'],
    })
    cleanlab_conf_mat_fig, cleanlab_cls_report = classification_evaluation(swapped_labels, cleanlab_error_mask)
    logger.experiment.log({"cleanlab_confusion_matrix": wandb.Image(cleanlab_conf_mat_fig)})
    logger.experiment.log({
        'cleanlab_accuracy': cleanlab_cls_report['accuracy'],
        'cleanlab_precision': cleanlab_cls_report['macro avg']['precision'],
        'cleanlab_recall': cleanlab_cls_report['macro avg']['recall'],
        'cleanlab_f1': cleanlab_cls_report['macro avg']['f1-score'],
    })

    print("Logging the results...")
    df = pd.DataFrame({
        "given_label": decoded_targets,
        "pred_label": decoded_preds,
        "swapped": swapped_labels,
        "label_issue": error_mask,
        "label_score": scores,
        "cleanlab_issue": cleanlab_error_mask,
        "cleanlab_score": cleanlab_scores,
        "images": [wandb.Image(image) for image in images]
    })
    logger.experiment.log({"result": wandb.Table(dataframe=df)})


@hydra.main(version_base=None, config_path="../configs", config_name="test_cla.yaml")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    data: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer)
    logger = hydra.utils.instantiate(cfg.logger)

    num_folds: int = cfg.get("num_folds", 4)
    
    accum_data = {'images': [], 'targets': [], 'probs': [], 'swapped_labels': []}

    for k in range(num_folds):
        data_module = data(fold_index=k)
        data_module.setup()
        model_instance = model(fold=k, num_classes=data_module.num_classes)
        callbacks = configure_callbacks(cfg, k)
        train_and_predict(model_instance, data_module, trainer, logger, callbacks, accum_data)

    targets, probs, preds, swapped_labels = process_accumulated_data(accum_data)
    evaluate_and_log(cfg, accum_data['images'], targets, probs, preds, swapped_labels, data_module, logger)

if __name__ == '__main__':
    main()