"""
Comet logger utilities.
"""
from __future__ import annotations
import json
from pathlib import Path

from pytorch_lightning.loggers import CometLogger
import hydra
from omegaconf import DictConfig


def get_comet_logger(lg_conf: DictConfig) -> CometLogger:
    """Get comet logger.

    Args:
        lg_conf (DictConfig):
            Arguments of `pytorch_lightning.loggers.CometLogger`.
            See details in https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.loggers.comet.html.  # noqa
    """
    filename: Path = Path("comet_key.json")
    comet_key: str | None = None

    # if a previous experiment exists, restart logging from the experiment.
    if filename.exists():
        with open(filename, "r") as fp:
            comet_key = json.load(fp)

    comet_logger = hydra.utils.instantiate(
        lg_conf,
        experiment_key=comet_key,
    )

    # dump experiment key for the next experiment.
    with open(filename, "w") as fp:
        json.dump(comet_logger.version, fp)

    return comet_logger
