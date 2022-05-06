from __future__ import annotations
import json
from pathlib import Path
from copy import deepcopy

import hydra
from omegaconf import DictConfig, OmegaConf


def get_wandb_logger(lg_conf: DictConfig):
    from pytorch_lightning.loggers import WandbLogger

    lg_conf = deepcopy(lg_conf)
    OmegaConf.set_struct(lg_conf, False)

    if lg_conf.get("version", None) is not None:
        raise ValueError("Do not use the version argument for WandbLogger.")

    run_id = lg_conf.pop("id", None)

    file = Path("wandb_key")
    if file.exists() and run_id is None:
        with open(file, "r") as fp:
            run_id = json.load(fp)

    wandb_logger: WandbLogger = hydra.utils.instantiate(lg_conf, id=run_id, resume="allow")

    with open(file, "w") as fp:
        json.dump(wandb_logger.version, fp)

    return wandb_logger
