import sys
sys.path.append("..")

import lightning.pytorch as pl 

import hydra 

def main():
    with hydra.initialize(version_base=None, config_path="../configs/model"):
        cfg = hydra.compose(config_name="efficientnet.yaml")
        model: pl.LightningModule = hydra.utils.instantiate(cfg)
        print(model(fold=0, num_classes=10))

        cfg = hydra.compose(config_name="retinanet")
        model: pl.LightningModule = hydra.utils.instantiate(cfg)
        print(model(fold=0, num_classes=10, trainable_backbone_layers=5))

        cfg = hydra.compose(config_name="fcn")
        model: pl.LightningModule = hydra.utils.instantiate(cfg)
        print(model(fold=0, num_classes=10))


if __name__ == '__main__':
    main()