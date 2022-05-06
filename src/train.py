from __future__ import annotations
import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers.base import LightningLoggerBase

from src.utils import utils
from src.loggers import get_logger

log = utils.get_logger(__name__)


def train(config: DictConfig) -> float:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """
    # Set seed for random number generators in pytorch, numpy and python.random
    seed_everything(config["seed"], workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        config.model,
        **getattr(datamodule, "model_kwargs", {}),
        _recursive_=False,
    )

    # Init lightning callbacks
    callbacks: list[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers and logger related callbacks.
    loggers: list[LightningLoggerBase] = []
    loggers, logger_callbacks = get_logger(config)
    callbacks.extend(logger_callbacks)

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    last_ckpt_path = Path("checkpoints", "last.ckpt")

    if last_ckpt_path.exists():
        resume_from_checkpoint = str(last_ckpt_path)

    elif config.trainer.get("resume_from_checkpoint"):
        resume_from_checkpoint = Path(config.trainer.resume_from_checkpoint)
        if not resume_from_checkpoint.exists():
            # reinterpret resume_from_checkpoint as relateve path
            resume_from_checkpoint = Path(config.work_dir, resume_from_checkpoint)
            if not resume_from_checkpoint.exists():
                log.error(f"Specified checkpoint {config.trainer.resume_from_checkpoint} is not found.")
                sys.exit(1)

    else:
        resume_from_checkpoint = None

    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=loggers,
        resume_from_checkpoint=resume_from_checkpoint,
        **getattr(datamodule, "trainer_kwargs", {}),
        **getattr(model, "trainer_kwargs", {}),
        _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    # Train the model
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    # Get metric score for hyperparameter optimization
    score = trainer.callback_metrics.get(config.get("optimized_metric"))

    # Evaluate model on test set, using the best model achieved during training
    if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
        log.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    # Print path to best checkpoint
    if not config.trainer.get("fast_dev_run"):
        log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    return score
