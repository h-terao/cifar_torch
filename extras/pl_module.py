from __future__ import annotations
from copy import deepcopy

import torch.nn as nn
import pytorch_lightning as pl


class EMA(pl.LightningModule):
    def __init__(
        self,
        momentum: float = 0.999,
        eman: bool = False,
    ) -> None:
        """
        Apply EMA to model parameters.

        Args:
            momentum (float): Momentum value for EMA.
            eman (bool): If True, apply EMA to BN parameters.
                         Reference: https://arxiv.org/abs/2101.08482.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.model = nn.Conv2d()
        self.ema_model = deepcopy(self.model)

    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0) -> None:
        
        def update_fn(main_iter, ema_iter, momentum) -> None:
            for (main_k, main_v), (ema_k, ema_v) in zip(main_iter, ema_iter):
                assert main_k == ema_k
                assert main_v.shape == ema_v.shape

                if "num_batches_tracked" in main_k:
                    ema_v.copy_(main_v)
                else:
                    ema_v.copy_(momentum * ema_v + (1 - momentum) * main_v)

        update_fn(self.model.named_parameters(), self.ema_model.named_parameters(), self.hparams.momentum)
        
        # 0 => copy
        momentum = self.hparams.momentum if self.hparams.eman else 0
        update_fn(self.model.named_buffers(), self.ema_model.named_buffers(), momentum)


class EstimateEpochLength(pl.LightningModule):
    #
    #  Estimate epochs from loaders.
    #
    @property
    def steps_per_epoch(self) -> int:
        """Estimate number of steps per epoch from trainer.
        Args:
            trainer (pl.Trainer): Trainer.

        Returns:
            Estimated number of steps per epoch.

        Reference:
            https://github.com/PyTorchLightning/pytorch-lightning/issues/5449
            https://github.com/Zasder3/train-CLIP/issues/29
        """
        limit_batches = self.trainer.limit_train_batches  # int or float.
        dataloader = self.trainer._data_connector._train_dataloader_source.dataloader()
        batches = len(dataloader)
        batches = (
            min(batches, limit_batches)
            if isinstance(limit_batches, int)
            else int(limit_batches * batches)
        )

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return batches // effective_accum

    @property
    def total_steps(self) -> int:
        """Estimate total steps from trainer.

        Args:
            trainer (pl.Trainer): Trainer.

        Returns:
            Estimated total steps.
        """
        if self.trainer.max_steps != -1:
            return self.trainer.max_steps
        else:
            return self.trainer.max_epochs * self.steps_per_epoch
