import os 
import json 

import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary
from sklearn.model_selection import StratifiedKFold, KFold
from torchvision.transforms import v2
from cleanlab import Datalab
import wandb

from lit_model import LitModel
from config import get_args, build_cfg
from utils import (
    swap_labels,
    roc_evaluation,
    plot_label_issue_examples, 
    plot_outlier_issues_examples, 
    plot_near_duplicate_issue_examples,
    label_issue_evaluation
)

def main(cfg):
    pl.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("medium")
    
    project_name = f"conf-learn-{cfg.dataset}"# f"{cfg.model}-{cfg.dataset}"
    logger = WandbLogger(
        project=project_name,
        log_model=False,
        save_dir="."
    )
    
    checkpoint_path = f"./checkpoints/{cfg.model}/{cfg.dataset}"
    callbacks = [
        # ModelCheckpoint(dirpath=checkpoint_path, save_top_k=2, monitor="valid-", mode="max"),
        ModelSummary(max_depth=-1),
        LearningRateMonitor(logging_interval='step')
    ]

    dataset = cfg.dataset_loaded
    def map_fn(item):
        transform = v2.Compose([
            v2.Resize(cfg.image_size, antialias=True),
            v2.Normalize(mean=cfg.image_mean, std=cfg.image_std),
        ])

        image = item.pop(cfg.image_key)
        
        image = image.float() / 255.0
        image = image.permute(2, 0, 1)
        image = transform(image)

        return {
            cfg.image_key: image,
            **item
        }
        
    fake_dataset = swap_labels(
        dataset,
        cfg.classes_to_swap,
        cfg.swap_prob,
        cfg.image_key, 
        cfg.label_key, 
        cfg.noisy_label_key
    )

    transformed_dataset = fake_dataset.map(map_fn, num_proc=cfg.num_workers)
    kfold = StratifiedKFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed)
    splitted_data = kfold.split(transformed_dataset[cfg.image_key], transformed_dataset[cfg.label_key])
    
    acc_val_idxs = []
    acc_val_embeddings = []
    acc_val_probs = []
    for fold, (train_idxs, val_idxs) in enumerate(splitted_data):
        model = LitModel(
            cfg.model,
            cfg.model_params,
            fold, 
            cfg.image_key, 
            cfg.label_key,
            cfg.optimizer,
            cfg.optimizer_params,
            cfg.scheduler,
            cfg.scheduler_params,
        )
        
        # subset the dataset
        train_dataset = transformed_dataset.select(train_idxs)
        val_dataset = transformed_dataset.select(val_idxs)
        
        # create dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, shuffle=True, **cfg.dataloader_params
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, shuffle=False, **cfg.dataloader_params
        )

        trainer = pl.Trainer(
            logger=logger, 
            max_epochs=cfg.num_epochs,
            precision=cfg.precision,
            benchmark=True,
            deterministic=True,
            callbacks=callbacks,
            num_sanity_val_steps=0,
        )
        
        # fit the model
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )
        # test the model, save the embeddings and out-of-sample probabilities
        out_of_sample_results = trainer.predict(
            model, 
            dataloaders=val_loader,
        )

        # accumulate the indices and results
        acc_val_idxs.extend(val_idxs)
        embeddings = torch.cat([result[0] for result in out_of_sample_results], dim=0)
        probs = torch.cat([result[1] for result in out_of_sample_results], dim=0)
        acc_val_embeddings.append(embeddings)
        acc_val_probs.append(probs)

    # for cleanlab to find the mislabeled samples
    features = torch.cat(acc_val_embeddings, dim=0).numpy()
    pred_probs = torch.cat(acc_val_probs, dim=0).numpy()
    indices = np.array(acc_val_idxs)

    # select the dataset from the original dataset (which will be used in the plotting)
    # weirdly, with_format("numpy" or "torch") causes an error in finding image_issues (e.g. dark, blurry, light, low_information ...)
    # Error in checking for image issues: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
    # so we use with_format(None) to avoid this error
    test_dataset = fake_dataset.select(indices).with_format(None)

    issue_types = {
        "label": {
            "clean_learning_kwargs": {
                # https://docs.cleanlab.ai/stable/cleanlab/filter.html#cleanlab.filter.find_label_issues 
                "find_label_issues_kwargs": {
                    # Method to determine which examples are flagged as label issues
                    # {'prune_by_class', 'prune_by_noise_rate', 'both', 'confident_learning', 'predicted_neq_given', 'low_normalized_margin', 'low_self_confidence'}
                    "filter_by": "confident_learning", 
                    # Which method to use to order the label issues by
                    # No need
                    # "rank_by_kwargs": {
                    #     # {"self_confidence", "normalized_margin", "confidence_weighted_entropy"}
                    #     "method": "normalized_margin"
                    # },
                    "n_jobs": None, # =1 for single core, None uses all cores 
                    "frac_noise": 1.0, # Used to only return top "frac_noise" * num_label_issues, 
                                       # Only applicable for filter_by=both|prune_by_class|prune_by_noise_rate 
                },
                # https://docs.cleanlab.ai/stable/cleanlab/rank.html#cleanlab.rank.get_label_quality_scores
                "label_quality_scores_kwargs": {
                    # Method to score the quality of the label (score=1: clean, score=0: noisy)
                    # {"self_confidence", "normalized_margin", "confidence_weighted_entropy"}
                    "method": "normalized_margin"
                }
            }
        },
        "outlier": {},
        "near_duplicate": {},
        "non_iid": {},
        "class_imbalance": {},
    }
    lab = Datalab(data=test_dataset, label_name=cfg.label_key, image_key=cfg.image_key)
    lab.find_issues(features=features, pred_probs=pred_probs, issue_types=issue_types)

    datalab_path = os.path.join(os.getcwd(), "lab_reports", cfg.model, cfg.dataset, logger.experiment.name)
    os.makedirs(datalab_path, exist_ok=True)
    
    # save the results to disk
    lab.save(datalab_path, force=True)
    
    aucroc, best_threshold, fig = roc_evaluation(lab)
    logger.log_metrics({"roc_auc": aucroc, "best_threshold": best_threshold})
    logger.experiment.log({"roc_curve": wandb.Image(fig)})
    fig.savefig(os.path.join(datalab_path, "roc_curve.png"))
    
    fig = plot_label_issue_examples(lab, 50)
    logger.experiment.log({"label_issue_examples": wandb.Image(fig)})
    fig.savefig(os.path.join(datalab_path, "label_issue_examples.png"))

    fig = plot_outlier_issues_examples(lab, 15)
    logger.experiment.log({"outlier_issue_examples": wandb.Image(fig)})
    fig.savefig(os.path.join(datalab_path, "outlier_issue_examples.png"))

    fig = plot_near_duplicate_issue_examples(lab, 15)
    if fig:
        logger.experiment.log({"near_duplicate_issue_examples": wandb.Image(fig)})
        fig.savefig(os.path.join(datalab_path, "near_duplicate_issue_examples.png"))

    fig, conf_mat_fig, cls_report = label_issue_evaluation(lab, 50)
    logger.experiment.log({"label_issue_evaluation": wandb.Image(fig)})
    fig.savefig(os.path.join(datalab_path, "label_issue_evaluation.png"))
    logger.experiment.log({"confusion_matrix": wandb.Image(conf_mat_fig)})
    conf_mat_fig.savefig(os.path.join(datalab_path, "confusion_matrix.png"))
    logger.experiment.log({
        'cls_report': cls_report,
        'accuracy': cls_report['accuracy'],
        'macro_avg_precision': cls_report['macro avg']['precision'],
        'macro_avg_recall': cls_report['macro avg']['recall'],
        'macro_avg_f1': cls_report['macro avg']['f1-score'],
        'weighted_avg_precision': cls_report['weighted avg']['precision'],
        'weighted_avg_recall': cls_report['weighted avg']['recall'],
        'weighted_avg_f1': cls_report['weighted avg']['f1-score']
    })
    with open(os.path.join(datalab_path, "cls_report.json"), "w") as f:
        json.dump(cls_report, f)


if __name__ == '__main__':
    args = get_args()
    cfg = build_cfg(args)
    main(cfg)