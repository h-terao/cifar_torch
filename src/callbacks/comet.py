from __future__ import annotations
import subprocess
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger, LoggerCollection
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only


def get_logger(trainer: Trainer) -> CometLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, CometLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, CometLogger):
                return logger

    raise Exception(
        "You are using comet related callback, but CometLogger was not found for some reason..."
    )


class UploadCode(Callback):
    """Upload all code files to comet, at the beginning of the run."""

    def __init__(self, code_dir: str, use_git: bool = True) -> None:
        """

        Args:
            code_dir: the code directory
            use_git: if using git, then upload all files that are not ignored by git.
            if not using git, then upload all '*.py' file
        """
        super().__init__()
        self.code_dir = code_dir
        self.use_git = use_git

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_logger(trainer=trainer)
        experiment = logger.experiment

        if self.use_git:
            # get .git folder
            # https://alexwlchan.net/2020/11/a-python-function-to-ignore-a-path-with-git-info-exclude/
            git_dir_path = Path(
                subprocess.check_output(["git", "rev-parse", "--git-dir"]).strip().decode("utf8")
            ).resolve()

            for path in Path(self.code_dir).resolve().rglob("*"):
                if (
                    path.is_file()
                    # ignore files in .git
                    and not str(path).startswith(str(git_dir_path))  # noqa: W503
                    # ignore files ignored by git
                    and (  # noqa: W503
                        subprocess.run(["git", "check-ignore", "-q", str(path)]).returncode == 1
                    )
                ):

                    experiment.log_code(file_name=str(path))

        else:
            for path in Path(self.code_dir).resolve().rglob("*.py"):
                experiment.log_code(file_name=str(path))


class UploadCheckpoints(Callback):
    """Upload checkpoints to wandb as an artifact, at the end of run."""

    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = True):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    @rank_zero_only
    def on_keyboard_interrupt(self, trainer, pl_module):
        self.on_train_end(trainer, pl_module)

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        logger = get_logger(trainer=trainer)
        experiment = logger.experiment

        if self.upload_best_only:
            experiment.log_model(
                name=trainer.checkpoint_callback.best_model_path,
                file_or_folder=trainer.checkpoint_callback.best_model_path,
                overwrite=True,
            )
        else:
            for path in Path(self.ckpt_dir).rglob("*.ckpt"):
                experiment.log_model(name=path.name, file_or_folder=str(path))
