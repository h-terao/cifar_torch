from __future__ import annotations

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import LightningLoggerBase
import hydra
from omegaconf import DictConfig

from src.utils import utils

log = utils.get_logger(__name__)


def get_logger(
    config: DictConfig,
) -> tuple[list[LightningLoggerBase], list[Callback]]:
    """Instantinates logger or logger related callbacks.

    Usage:
        get_logger(config.logger)
    """
    loggers = []
    logger_callbacks = []

    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                if "CometLogger" in lg_conf["_target_"]:
                    # CometLogger should be
                    from .comet_logger import get_comet_logger

                    item = get_comet_logger(lg_conf)
                elif "WandbLogger" in lg_conf["_target_"]:
                    from .wandb_logger import get_wandb_logger

                    item = get_wandb_logger(lg_conf)
                else:
                    item = hydra.utils.instantiate(lg_conf)

                if isinstance(item, LightningLoggerBase):
                    loggers.append(item)
                elif isinstance(item, Callback):
                    logger_callbacks.append(item)
                else:
                    raise ValueError(
                        (
                            "Only LightningLogger and LightningCallback are "
                            "allowed in `config.logger`."
                        )
                    )

                log.info(f"Instantiating logger <{lg_conf._target_}>")

    return loggers, logger_callbacks
