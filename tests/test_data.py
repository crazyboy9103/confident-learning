import sys
sys.path.append("..")

import lightning.pytorch as pl 

import hydra 
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

def main():
    with hydra.initialize(version_base=None, config_path="../configs/data"):
        # test_noisy_imagefolder("/datasets/conflearn/cla/applied_materials_processed")

        # for noise_type in ["overlook", "badloc", "swap"]:
        #     test_noisy_cocodetection("/datasets/conflearn/seg/battery", noise_type)
        # test_noisy_imagefolder_data("/datasets/conflearn/cla/ramen_processed_data")
        test_coco_detection("/datasets/conflearn/det/chocoball")
def test_coco_detection(data_root_dir: str):
    cfg = hydra.compose(config_name="coco", overrides=[f"root={data_root_dir}/images", f"annFile={data_root_dir}/label.json"])
    num_folds = 4
    data: pl.LightningDataModule = hydra.utils.instantiate(cfg, num_folds=num_folds)
    
    datamodules = [data(fold_index=i) for i in range(num_folds)]

    for datamodule in datamodules:
        datamodule.setup()
        loader = datamodule.train_dataloader()
        for image, target in loader:
            for ann in target:
                x1, y1, x2, y2 = ann["boxes"].unbind(1)
                if (x1 == x2).any() or (y1 == y2).any():
                    print("Bad box", ann)
        break

def test_noisy_imagefolder_data(data_root_dir: str):
    cfg = hydra.compose(config_name="imagefolder", overrides=[f"root={data_root_dir}"])
    num_folds = 4
    data: pl.LightningDataModule = hydra.utils.instantiate(cfg, num_folds=num_folds)
    
    datamodules = [data(fold_index=i) for i in range(num_folds)]

    for datamodule in datamodules:
        datamodule.setup()
        images = datamodule.pred_images()
        print(len(images))
        print(len([i for i in datamodule.data_pred]))
        break


def test_noisy_imagefolder(data_root_dir: str):
    cfg = hydra.compose(config_name="imagefolder", overrides=[f"root={data_root_dir}"])
    num_folds = 4
    data: pl.LightningDataModule = hydra.utils.instantiate(cfg, num_folds=num_folds)
    
    datamodules = [data(fold_index=i) for i in range(num_folds)]
    
    orig_labels = []
    noisy_labels = []

    train_intersect = set()
    val_intersect = set()

    for datamodule in datamodules:
        datamodule.setup()
        assert set(datamodule.data_train.idxs) & set(datamodule.data_val.idxs) == set()
        if not train_intersect:
            train_intersect = set(datamodule.data_train.idxs)
        else:
            train_intersect &= set(datamodule.data_train.idxs)

        if not val_intersect:
            val_intersect = set(datamodule.data_val.idxs)
        else:
            val_intersect &= set(datamodule.data_val.idxs)

        pred_info = datamodule.predict_info()

        orig_labels.extend(pred_info["orig_labels"])
        noisy_labels.extend(pred_info["noisy_labels"])
    
    assert train_intersect == set() and val_intersect == set()

    conf = confusion_matrix(orig_labels, noisy_labels, normalize='true')
    disp = ConfusionMatrixDisplay(conf, display_labels=range(1, datamodule.num_classes + 1))
    disp.plot()
    plt.savefig(f"{data_root_dir.split('/')[-1]}.png")

def test_noisy_cocodetection(data_root_dir, noise_type):
    def check_noisy_label_distribution(dataloader):
        dist = {}
        for _, targets in dataloader:
            for target in targets:
                for ann in target:
                    category_id = ann["category_id"]
                    noisy_label = ann["noisy_label"]
                    dist.setdefault(category_id, []).append(noisy_label)
        return dist
        
    overrides = [f"root={data_root_dir}/images", f"annFile={data_root_dir}/label.json", f"noise_type={noise_type}"]

    if noise_type == "overlook":
        overrides.append("noise_config._target_=src.data.cocodetection.OverlookNoiseConfig")
        overrides.append("+noise_config.prob=0.1")
        overrides.append("+noise_config.debug=True") # this is required for testing, 
                                                    # otherwise overlooked anns would not be visible 
    elif noise_type == "badloc":
        overrides.append("noise_config._target_=src.data.cocodetection.BadLocNoiseConfig")
        overrides.append("+noise_config.prob=0.1")
        overrides.append("+noise_config.max_pixel=20")
    
    elif noise_type == "swap":
        overrides.append("noise_config._target_=src.data.cocodetection.SwapNoiseConfig")
        overrides.append("+noise_config.num_classes_to_swap=3")
        overrides.append("+noise_config.prob=0.1")
    
    else:
        raise ValueError(f"Unknown noise type {noise_type}")
    cfg = hydra.compose(config_name="coco", overrides=overrides)
    num_folds = 4

    data: pl.LightningDataModule = hydra.utils.instantiate(cfg, num_folds=num_folds)
    datamodules = [data(fold_index=i) for i in range(num_folds)]

    train_dist = {}
    test_dist = {}
    for datamodule in datamodules:
        datamodule.setup()
        
        train_dist_k = check_noisy_label_distribution(datamodule.train_dataloader())
        test_dist_k = check_noisy_label_distribution(datamodule.predict_dataloader())

        for i in range(1, datamodule.num_classes + 1):
            train_list = train_dist_k.get(i, [])
            test_list = test_dist_k.get(i, [])

            train_dist.setdefault(i, []).extend(train_list)
            test_dist.setdefault(i, []).extend(test_list)
    
    for i in range(1, datamodule.num_classes + 1):
        train_dist[i] = round(sum(train_dist[i]) / len(train_dist[i]), 2) if train_dist[i] else 0
        test_dist[i] = round(sum(test_dist[i]) / len(test_dist[i]), 2) if test_dist[i] else 0

    print("noise_type", noise_type)
    print("train_dist", train_dist)
    print("test_dist", test_dist)


if __name__ == '__main__':
    main()