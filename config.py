import argparse

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, CosineAnnealingLR
from datasets import load_dataset # https://huggingface.co/docs/datasets/create_dataset

from utils import calculate_mean_std
from scheduler import LinearWarmUpMultiStepDecay, CosineAnnealingWarmUpRestarts

def get_args():
    parser = argparse.ArgumentParser()
    dataset_group = parser.add_argument_group('Dataset')
    dataset_group.add_argument('--dataset', type=str, default='covid', choices=['cifar10', 'cifar100', 'fashion_mnist', 'mnist', 'fruit', 'pill', 'pet', 'car', 'covid', 'volvo', 'fetus', 'ramen'], help="Dataset to load using huggingface api, imagefolder is used to load custom dataset")
    dataset_group.add_argument('--data_dir', type=str, default='/datasets/Fruit_processed', help="Directory to custom dataset, must be set if dataset is imagefolder, ignored otherwise")
    dataset_group.add_argument('--split', type=str, default="all", help="Split to use from the dataset, https://huggingface.co/docs/datasets/v2.17.0/loading#slice-splits")
    dataset_group.add_argument('--image_size', type=int, default=256)

    dataloader_group = parser.add_argument_group('Dataloader')
    dataloader_group.add_argument('--batch_size', type=int, default=64)
    dataloader_group.add_argument('--num_workers', type=int, default=8)

    model_group = parser.add_argument_group('Model')
    model_group.add_argument('--model', type=str, default='resnet18', help='Backbone model from timm.create_model')
    model_group.add_argument('--pretrained', action='store_true', default=True)

    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--num_epochs', type=int, default=12)
    train_group.add_argument('--folds', type=int, default=4)
    train_group.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'])
    train_group.add_argument('--scheduler', type=str, default='CosineAnnealingLR', choices=['CosineAnnealingWarmUpRestarts', 'CosineAnnealingWarmRestarts', 'LinearWarmUpMultiStepDecay', 'CosineAnnealingLR'])
    train_group.add_argument('--learning_rate', type=float, default=0.001)
    train_group.add_argument('--precision', type=str, default='32', choices=['bf16', 'bf16-mixed', '16', '16-mixed', '32', '32-true', '64', '64-true'])

    # Misc
    parser.add_argument('--wandb', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=2024)
    args = parser.parse_args()
    return args

def build_cfg(cfg):
    """
    Build configuration from command line arguments
    """
    # alter hparams based on dataset
    match cfg.dataset:
        case "cifar10":
            cfg.image_size = 32
            cfg.batch_size = 128
            cfg.num_epochs = 12
            cfg.folds = 4
            cfg.optimizer = "Adam"
            cfg.learning_rate = 0.001

            cfg.classes_to_swap = [("cat", "dog"), ("truck", "automobile"), ("deer", "horse")]
            cfg.swap_prob = [0.2, 0.2, 0.2]

        case "cifar100":
            pass
        case "fashion_mnist":
            raise NotImplementedError("Fashion MNIST is not implemented yet")
        case "mnist":
            raise NotImplementedError("MNIST is not implemented yet")
        case "fruit":
            cfg.image_size = 256
            cfg.batch_size = 32
            cfg.num_epochs = 12
            cfg.folds = 4
            cfg.optimizer = "Adam"
            cfg.learning_rate = 0.001 
            cfg.data_dir = "/datasets/Fruit_processed"

            cfg.classes_to_swap = [("Apple_Fresh", "Apple_Rotten"), ("Banana_Fresh", "Banana_Rotten"), ("Orange_Fresh", "Orange_Rotten")]
            cfg.swap_prob = [0.1, 0.1, 0.1]

        case "pill":
            cfg.image_size = 256
            cfg.batch_size = 32
            cfg.num_epochs = 12
            cfg.folds = 4
            cfg.optimizer = "Adam"
            cfg.learning_rate = 0.001 
            cfg.data_dir = "/datasets/applied_materials_processed"

            cfg.classes_to_swap = [("chipping", "good"), ("chipping", "picking"), ("good", "picking")]
            cfg.swap_prob = [0.1, 0.1, 0.1]

        case "pet":
            cfg.image_size = 256
            cfg.batch_size = 32
            cfg.num_epochs = 12
            cfg.folds = 4
            cfg.optimizer = "Adam"
            cfg.learning_rate = 0.001 
            cfg.data_dir = "/datasets/pet_images_processed"

            cfg.classes_to_swap = [("Sphynx", "Russian_Blue"), ("Persian", "Ragdoll"), ("chihuahua", "japanese_chin")]
            cfg.swap_prob = [0.1, 0.1, 0.1]

        case "car":
            cfg.image_size = 256
            cfg.batch_size = 32
            cfg.num_epochs = 12
            cfg.folds = 4
            cfg.optimizer = "Adam"
            cfg.learning_rate = 0.001 
            cfg.data_dir = "/datasets/cars_processed"
            # arbitrary.. dont know cars well
            cfg.classes_to_swap = [("1", "130"), ("5", "26"), ("7", "8")]
            cfg.swap_prob = [0.1, 0.1, 0.1]
        
        case "covid":
            cfg.image_size = 256
            cfg.batch_size = 32
            cfg.num_epochs = 12
            cfg.folds = 4
            cfg.optimizer = "Adam"
            cfg.learning_rate = 0.001 
            cfg.data_dir = "/datasets/085.Covid19_3Class_317ea_Kaggle/Covid19-dataset"
            cfg.classes_to_swap = [("Covid", "Normal"), ("Covid", "Viral Pneumonia"), ("Normal", "Viral Pneumonia")]
            cfg.swap_prob = [0.1, 0.1, 0.1]
        
        case "volvo":
            cfg.image_size = 256
            cfg.batch_size = 32
            cfg.num_epochs = 12
            cfg.folds = 4
            cfg.optimizer = "Adam"
            cfg.learning_rate = 0.001 
            cfg.data_dir = "/datasets/111.볼보_3class_861ea/VolvoWeldingSoundTest"
            cfg.classes_to_swap = [("Defect", "Noise"), ("Defect", "Normal"), ("Noise", "Normal")]
            cfg.swap_prob = [0.1, 0.1, 0.1]
        
        case "fetus":
            cfg.image_size = 256
            cfg.batch_size = 32
            cfg.num_epochs = 12
            cfg.folds = 4
            cfg.optimizer = "Adam"
            cfg.learning_rate = 0.001
            cfg.data_dir = "/datasets/286. self_태아초음파/1. 수신 Data/Fetal Ultra Sound/processed_data"
            cfg.classes_to_swap = [("Trans-cerebellum", "Trans-thalamic"), ("Trans-cerebellum", "Trans-ventricular"), ("Trans-thalamic", "Trans-ventricular")]
            cfg.swap_prob = [0.1, 0.1, 0.1]

        case "ramen":
            cfg.image_size = 256
            cfg.batch_size = 32
            cfg.num_epochs = 12
            cfg.folds = 4
            cfg.optimizer = "Adam"
            cfg.learning_rate = 0.001
            cfg.data_dir = "/datasets/296. 농심_라면후레이크/processed_data"
            cfg.classes_to_swap = [("normal", "hair"), ("thread", "hair"), ("normal", "vinyl")]
            cfg.swap_prob = [0.1, 0.1, 0.1]

        case _:
            raise ValueError(f"Invalid dataset {cfg.dataset}")
    
    # Dataset
    if cfg.dataset in ['cifar10', 'cifar100', 'fashion_mnist', 'mnist']:
        cfg.dataset_loaded = load_dataset(cfg.dataset, split=cfg.split).with_format("torch")
    else:
        cfg.dataset_loaded = load_dataset('imagefolder', data_dir=cfg.data_dir, split=cfg.split).with_format("torch")

    # different datasets have different keys for images (and possibly labels too)
    # This is a bit fragile, but it works for now
    image_key, label_key = cfg.dataset_loaded.column_names
    cfg.image_key = image_key
    cfg.label_key = label_key
    cfg.num_classes = len(cfg.dataset_loaded.features[label_key].names) 

    if cfg.pretrained:
        cfg.image_mean = (0.485, 0.456, 0.406)
        cfg.image_std = (0.229, 0.224, 0.225) 
    elif cfg.dataset == 'cifar10':
        cfg.image_mean = (0.4914, 0.4822, 0.4465)
        cfg.image_std = (0.2023, 0.1994, 0.2010)
    elif cfg.dataset == 'cifar100':
        cfg.image_mean = (0.5071, 0.4867, 0.4408)
        cfg.image_std = (0.2675, 0.2565, 0.2761)
    else:
        # calculate mean and std from the dataset
        cfg.image_mean, cfg.image_std = calculate_mean_std(cfg.dataset_loaded, image_key=cfg.image_key)

    cfg.image_size = (cfg.image_size, cfg.image_size)
    cfg.noisy_label_key = "noisy_label"
    
    # DataLoader 
    cfg.dataloader_params = {
        "batch_size": cfg.batch_size, 
        "num_workers": cfg.num_workers,
        "pin_memory": True
    }
    # Optimization
    cfg.optimizer = torch.optim.__dict__[cfg.optimizer]

    if cfg.optimizer == torch.optim.SGD:
        cfg.optimizer_params = {
            "lr": cfg.learning_rate, 
            "momentum": 0, 
            "weight_decay": 1e-4, 
            "nesterov": False
        }

    elif cfg.optimizer == torch.optim.Adam:
        cfg.optimizer_params = {
            "lr": cfg.learning_rate, 
            "betas": (0.9, 0.999), 
            "eps": 1e-08, 
            "weight_decay": 1e-4, 
            "amsgrad": False
        }

    cfg.scheduler = torch.optim.lr_scheduler.__dict__[cfg.scheduler]
    steps_per_epoch = len(cfg.dataset_loaded) // cfg.batch_size * (cfg.folds - 1) // cfg.folds
    if cfg.scheduler == LinearWarmUpMultiStepDecay:
        cfg.scheduler_params = {
            "warmup_start_lr": cfg.learning_rate, 
            "milestones": [steps_per_epoch * (cfg.num_epochs - 2), steps_per_epoch * (cfg.num_epochs - 1)], 
            "warmup_steps": steps_per_epoch * 2
        }

    elif cfg.scheduler == CosineAnnealingWarmRestarts:
        cfg.scheduler_params = {
            "T_0": steps_per_epoch * cfg.num_epochs, 
            "T_mult": 1, 
            "eta_min": 0, 
            "last_epoch": -1, 
            "verbose": False
        }
    
    elif cfg.scheduler == CosineAnnealingLR:
        cfg.scheduler_params = {
            "T_max": steps_per_epoch * 4, 
            "eta_min": 0, 
            "last_epoch": -1, 
            "verbose": False
        }

    elif cfg.scheduler == CosineAnnealingWarmUpRestarts:
        cfg.scheduler_params = {
            "T_0": steps_per_epoch * cfg.num_epochs, 
            "T_mult": 1, 
            "eta_max": cfg.learning_rate, 
            "T_up": steps_per_epoch * 2, 
            "gamma": 1., 
            "last_epoch": -1
        }

    # Model
    cfg.model_params = {
        "pretrained": cfg.pretrained, 
        "num_classes": cfg.num_classes
    }

    return cfg
    

if __name__ == "__main__":
    print(build_cfg(get_args()))

    