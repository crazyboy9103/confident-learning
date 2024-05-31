import sys
sys.path.append("..")
from itertools import chain

import torch
import lightning.pytorch as pl 
import hydra 
from omegaconf import DictConfig
import pandas as pd
import wandb 
import numpy as np
from cleanlab.object_detection.rank import get_label_quality_scores, compute_badloc_box_scores, compute_overlooked_box_scores, compute_swap_box_scores

from src.conflearn.detection import Detection
from src.conflearn.utils import softmin1d_pooling
from src.eval_utils import roc_evaluation

def prepare_for_conflearn(target_dict, pred_dict):
    for k, v in pred_dict.items():
        if isinstance(v, torch.Tensor):
            pred_dict[k] = v.cpu().numpy()
    
    for k, v in target_dict.items():
        if isinstance(v, torch.Tensor):
            target_dict[k] = v.cpu().numpy()

def prepare_for_cleanlab(targets, preds, num_classes):
    cleanlab_labels = []
    cleanlab_preds = [[[] for _ in range(num_classes)] for _ in range(len(preds))]

    for target in targets:
        cleanlab_labels.append({
            "bboxes": target["boxes"],
            "labels": target["labels"]
        })
    
    for i, pred in enumerate(preds):
        for class_idx in range(num_classes):
            class_mask = pred["labels"] == (class_idx + 1)
            pred_boxes = pred["boxes"][class_mask]
            pred_scores = pred["scores"][class_mask]

            if len(class_mask) == 0:
                continue

            class_predictions = np.hstack((pred_boxes, pred_scores[:, None]))

            cleanlab_preds[i][class_idx].append(class_predictions)

    for pred in cleanlab_preds:
        for i, p in enumerate(pred):
            pred[i] = np.array(p[0]) if len(p) > 0 else np.zeros((0, 5))

    return cleanlab_labels, cleanlab_preds

def parse_boxes_for_wandb(target_boxes_per_image, target_labels_per_image, pred_boxes_per_image, pred_labels_per_image, pred_scores_per_image, class_labels):
    # explicitly cast to int and float to avoid serialization issues
    predictions = {
        "box_data": [
            {
                "position": {
                    "minX": int(pred_box[0]),
                    "minY": int(pred_box[1]),
                    "maxX": int(pred_box[2]),
                    "maxY": int(pred_box[3])
                }, 
                "class_id": int(pred_label),
                "domain": "pixel",
                "scores": {"score": float(pred_score)}
            } for pred_box, pred_label, pred_score in zip(pred_boxes_per_image, pred_labels_per_image, pred_scores_per_image)
        ], 
        "class_labels": class_labels
    }

    ground_truths = {
        "box_data": [
            {
                "position": {
                    "minX": int(target_box[0]),
                    "minY": int(target_box[1]),
                    "maxX": int(target_box[2]),
                    "maxY": int(target_box[3]),
                },
                "class_id": int(target_label),
                "domain": "pixel"
            } for target_box, target_label in zip(target_boxes_per_image, target_labels_per_image)
        ], 
        "class_labels": class_labels
    }
    return {
        "predictions": predictions,
        "ground_truths": ground_truths
    }
    
@hydra.main(version_base=None, config_path="../configs", config_name="test_det.yaml")
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
                callback = callback(filename=f"det_model_{cfg.task_name}_{k}", monitor=f"valid_map_{k}", mode="max")

            callbacks.append(callback)

        trainer_instance = trainer(logger=logger, callbacks=callbacks)
        trainer_instance.fit(model_instance, datamodule=data_module)
        preds = trainer_instance.predict(model_instance, datamodule=data_module, ckpt_path="best")
        
        targets = [result[0] for result in preds]
        preds = [result[1] for result in preds]

        targets = list(chain.from_iterable(targets))
        preds = list(chain.from_iterable(preds))

        accum_targets.extend(targets)
        accum_preds.extend(preds)

        images = data_module.pred_images()
        accum_images.extend(images)

    for target, pred in zip(accum_targets, accum_preds):
        prepare_for_conflearn(target, pred)

    conflearn = Detection(accum_targets, accum_preds)
    badloc_scores, overlooked_scores, swapped_scores = conflearn.get_result(cfg.alpha, cfg.badloc_min_confidence, cfg.min_confidence, cfg.pooling, cfg.softmin_temperature)
    pooled_scores = [(s1 * s2 * s3) ** (1/3) for s1, s2, s3 in zip(badloc_scores, overlooked_scores, swapped_scores)]

    cleanlab_labels, cleanlab_preds = prepare_for_cleanlab(accum_targets, accum_preds, data_module.num_classes)
    cleanlab_scores = get_label_quality_scores(cleanlab_labels, cleanlab_preds)
    
    cleanlab_badloc_scores = compute_badloc_box_scores(labels=cleanlab_labels, predictions=cleanlab_preds)
    cleanlab_overlooked_scores = compute_overlooked_box_scores(labels=cleanlab_labels, predictions=cleanlab_preds)
    cleanlab_swapped_scores = compute_swap_box_scores(labels=cleanlab_labels, predictions=cleanlab_preds)
    cleanlab_badloc_scores = [softmin1d_pooling(score, temperature=cfg.softmin_temperature) for score in cleanlab_badloc_scores]
    cleanlab_overlooked_scores = [softmin1d_pooling(score, temperature=cfg.softmin_temperature) for score in cleanlab_overlooked_scores]
    cleanlab_swapped_scores = [softmin1d_pooling(score, temperature=cfg.softmin_temperature) for score in cleanlab_swapped_scores]

    # Consider the image is noisy if any of the annotations for that image is noisy 
    # otherwise, the image is normal 
    noisy_labels = [any(target["noisy_labels"]) for target in accum_targets]

    # class_map = {1: {"id": 1, "name": "person", "supercategory": "person"...}, 2: {"id": 2, "name": "car", "supercategory": "vehicle"...}, ...}
    class_map = data_module.dataset.classes 

    class_labels = {class_id: cat["name"] for class_id, cat in class_map.items()}

    target_boxes = [target["boxes"] for target in accum_targets]
    target_labels = [target["labels"] for target in accum_targets]
    
    pred_boxes = [pred["boxes"] for pred in accum_preds]
    pred_labels = [pred["labels"] for pred in accum_preds]
    pred_scores = [pred["scores"] for pred in accum_preds]

    class_set = wandb.Classes([
        {"name": name, "id": class_id} for class_id, name in class_labels.items()
    ])

    df = pd.DataFrame({
        "swapped": noisy_labels,
        "label_score": pooled_scores,
        "cleanlab_score": cleanlab_scores,
        "badloc_score": badloc_scores,
        "overlooked_score": overlooked_scores,
        "swapped_score": swapped_scores,
        "cleanlab_badloc_score": cleanlab_badloc_scores,
        "cleanlab_overlooked_score": cleanlab_overlooked_scores,
        "cleanlab_swapped_score": cleanlab_swapped_scores,
        "images": [wandb.Image(
            accum_images[i],
            boxes = parse_boxes_for_wandb(target_boxes[i], target_labels[i], pred_boxes[i], pred_labels[i], pred_scores[i], class_labels),
            classes=class_set
        ) for i in range(len(accum_images))]
    })
    logger.experiment.log({"result": wandb.Table(dataframe=df)})

    aucroc, best_threshold, roc_curve_fig = roc_evaluation(noisy_labels, pooled_scores)

    logger.experiment.log({
        'aucroc': aucroc,
        'best_threshold': best_threshold,
        'roc_curve': wandb.Image(roc_curve_fig)
    })

    cleanlab_aucroc, cleanlab_best_threshold, cleanlab_roc_curve_fig = roc_evaluation(noisy_labels, cleanlab_scores)

    logger.experiment.log({
        'cleanlab_aucroc': cleanlab_aucroc,
        'cleanlab_best_threshold': cleanlab_best_threshold,
        'cleanlab_roc_curve': wandb.Image(cleanlab_roc_curve_fig)
    })




if __name__ == '__main__':
    main()