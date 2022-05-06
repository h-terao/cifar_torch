from __future__ import annotations
from contextlib import contextmanager

import torch
import torch.nn as nn


@contextmanager
def freeze_bn(model: nn.Module):
    """
    Disable BN layers to track stats of their inputs.
    Inspired from Bn_Controller in https://github.com/TorchSSL/TorchSSL/blob/main/train_utils.py
    Args:
        model (nn.Module):
            Target model to freeze the BN layers.
    Example:
        >>> net(x_1)  # stats of the hidden states from x_1 is tracked by BN layers.
        >>> with freeze_bn(net):
                net(x_2)  # stats of the hidden states from x_2 is not tracked by BN layers.
    """

    backup = {}
    for k, m in model.named_modules():
        if isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            backup[k] = {
                "running_mean": m.running_mean.data.clone(),
                "running_var": m.running_var.data.clone(),
                "num_batches_tracked": m.num_batches_tracked.data.clone(),
            }

    yield

    # Restore the previous parameters.
    for k, m in model.named_modules():
        if isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.running_mean.data = backup[k]["running_mean"]
            m.running_var.data = backup[k]["running_var"]
            m.num_batches_tracked.data = backup[k]["num_batches_tracked"]


class Timer:
    """
    Timer class for CUDA computation in PyTorch.
    Example:
        timer = Timer()
        with timer.time():
            # do heavy computation with CUDA.
        timer.get_elapsed_time()  # get elapsed time (ms).
        timer.get_elapsed_time(unit="s")  # get elapsed time (sec).
    """

    def __init__(self) -> None:
        self.reset()

    @contextmanager
    def time(self) -> None:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()

        yield

        end_event.record()
        torch.cuda.synchronize()

        self.elapsed_time = start_event.elapsed_time(end_event)

    def reset(self):
        self.elapsed_time = None

    def get_elapsed_time(self, unit: str = "ms") -> float | None:
        assert unit in ["ms", "s", "m", "h"]

        if self.elapsed_time is None:
            return None

        elif unit == "ms":
            return self.elapsed_time

        elif unit == "s":
            return self.elapsed_time * 1000

        elif unit == "m":
            return self.elapsed_time * 1000 * 60

        elif unit == "h":
            return self.elapsed_time * 1000 * 60 * 60

        else:
            raise NotImplementedError()
