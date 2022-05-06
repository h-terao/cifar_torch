from __future__ import annotations
from copy import deepcopy
from typing import Any

import torch
import torch.nn.functional as F
import timm
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
import hydra
from omegaconf import DictConfig, OmegaConf


class CifarLitModel(pl.LightningModule):
    """Sample model.

    Args:
        optimizer (DictConfig): Optimizer config.
        scheduler (DictConfig | None): LR scheduler config. If None, do not set lr_scheduler.
    """

    def __init__(
        self,
        arch: str,
        n_classes: int,
        pretrained: bool,
        optimizer: DictConfig,
        lr_scheduler: DictConfig | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.model = timm.create_model(arch, pretrained, num_classes=n_classes)

    def shared_step(self, batch: Any, prefix: str) -> torch.Tensor:
        input, target = batch
        n = len(input)

        logit = self.model(input)
        loss = F.cross_entropy(logit, target)

        self.log_dict(
            {
                f"{prefix}/loss": loss,
                f"{prefix}/acc": accuracy(logit, target),
            },
            batch_size=n,
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer,
            params=self.parameters(),
        )

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            lr_scheduler = deepcopy(self.hparams.lr_scheduler)
            OmegaConf.set_struct(lr_scheduler, False)

            scheduler = hydra.utils.instantiate(lr_scheduler.pop("scheduler"), optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": dict(
                    scheduler=scheduler,
                    **OmegaConf.to_container(lr_scheduler),
                ),
            }
