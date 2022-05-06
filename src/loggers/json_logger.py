from __future__ import annotations
import json
import shutil
from copy import deepcopy
from tempfile import TemporaryDirectory
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning import (
    Callback,
    Trainer,
    LightningModule,
)


class JSONLogger(Callback):
    """A statefull json logger. Unlike the native lightning loggers, this logger stores observations in checkpoints."""

    def __init__(self, name: str, save_dir: str):
        self.save_dir = save_dir
        self.name = name
        self._log = []

        # create save_dir.
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.log_metrics(trainer)

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.log_metrics(trainer, filter_key="test")

    def on_load_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        callback_state: dict[str, Any],
    ) -> None:
        self._log = json.loads(callback_state["log"])

    def on_save_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: dict[str, Any],
    ) -> dict:
        return {"log": json.dumps(self._log)}

    @rank_zero_only
    def log_metrics(self, trainer: "pl.Trainer", filter_key: str | None = None):
        metrics = deepcopy(trainer.logged_metrics)
        if metrics:
            if filter_key is None:
                metrics = {k: float(v) for k, v in metrics.items()}
            else:
                metrics = {k: float(v) for k, v in metrics.items() if filter_key in k}

            metrics.update(epoch=trainer.current_epoch, step=trainer.global_step)
            self._log.append(metrics)

            with TemporaryDirectory(dir=self.save_dir) as tmp_dir:
                path = Path(tmp_dir, self.name)
                with open(path, "w") as fp:
                    json.dump(self._log, fp, indent=2)

                new_path = Path(self.save_dir, self.name)
                shutil.move(path, new_path)
