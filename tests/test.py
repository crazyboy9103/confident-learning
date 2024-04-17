import lightning.pytorch as pl 

import hydra 
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../configs", config_name="test.yaml")
def main(cfg: DictConfig):
    callbacks = []

    for k, v in cfg.callbacks.items():
        callback: pl.Callback = hydra.utils.instantiate(v)

        if k == "model_checkpoint":
            callback = callback(filename = "model_checkpoint", monitor="val_loss", mode="min")

        callbacks.append(callback)


if __name__ == '__main__':
    main()