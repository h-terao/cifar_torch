"""
Print metrics by chainer like method
Modify from:
https://github.com/pfnet/pytorch-pfn-extras/blob/v0.2.0/pytorch_pfn_extras/training/extensions/print_report.py

Note that using this callback with RichProgressbar may breaks table.
"""
from __future__ import annotations
import os
import json
from copy import deepcopy
from typing import Any, Callable

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning import (
    Callback,
    Trainer,
    LightningModule,
)
import tqdm

# Code from
# https://github.com/pfnet/pytorch-pfn-extras/blob/v0.2.0/pytorch_pfn_extras/training/extensions/util.py
if os.name == "nt":
    import ctypes

    _STD_OUTPUT_HANDLE = -11

    class _COORD(ctypes.Structure):
        _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]

    class _SMALL_RECT(ctypes.Structure):
        _fields_ = [
            ("Left", ctypes.c_short),
            ("Top", ctypes.c_short),
            ("Right", ctypes.c_short),
            ("Bottom", ctypes.c_short),
        ]

    class _CONSOLE_SCREEN_BUFFER_INFO(ctypes.Structure):
        _fields_ = [
            ("dwSize", _COORD),
            ("dwCursorPosition", _COORD),
            ("wAttributes", ctypes.c_ushort),
            ("srWindow", _SMALL_RECT),
            ("dwMaximumWindowSize", _COORD),
        ]

    def set_console_cursor_position(x, y):
        """Set relative cursor position from current position to (x,y)"""

        whnd = ctypes.windll.kernel32.GetStdHandle(_STD_OUTPUT_HANDLE)
        csbi = _CONSOLE_SCREEN_BUFFER_INFO()
        ctypes.windll.kernel32.GetConsoleScreenBufferInfo(whnd, ctypes.byref(csbi))
        cur_pos = csbi.dwCursorPosition
        pos = _COORD(cur_pos.X + x, cur_pos.Y + y)
        ctypes.windll.kernel32.SetConsoleCursorPosition(whnd, pos)

    def erase_console(x, y, mode=0):
        """Erase screen.
        Mode=0: From (x,y) position down to the bottom of the screen.
        Mode=1: From (x,y) position down to the beginning of line.
        Mode=2: Hole screen
        """

        whnd = ctypes.windll.kernel32.GetStdHandle(_STD_OUTPUT_HANDLE)
        csbi = _CONSOLE_SCREEN_BUFFER_INFO()
        ctypes.windll.kernel32.GetConsoleScreenBufferInfo(whnd, ctypes.byref(csbi))
        cur_pos = csbi.dwCursorPosition
        wr = ctypes.c_ulong()
        if mode == 0:
            num = csbi.srWindow.Right * (csbi.srWindow.Bottom - cur_pos.Y) - cur_pos.X
            ctypes.windll.kernel32.FillConsoleOutputCharacterA(
                whnd, ord(" "), num, cur_pos, ctypes.byref(wr)
            )
        elif mode == 1:
            num = cur_pos.X
            ctypes.windll.kernel32.FillConsoleOutputCharacterA(
                whnd, ord(" "), num, _COORD(0, cur_pos.Y), ctypes.byref(wr)
            )
        elif mode == 2:
            os.system("cls")


def always_true(*args, **kwargs) -> bool:
    return True


# Copy from pytorch-pfn-extras
def create_header_and_templates(
    entries: list[str],
) -> tuple[str, list[tuple[str, str, str]]]:
    """Construct header and templates from `entries`
    Args:
        entries (list): list of str
    Returns:
        header (str): header string
        templates (str): template string for print values.
    """
    # format information
    entry_widths = [max(10, len(s)) for s in entries]

    header = "  ".join(("{:%d}" % w for w in entry_widths)).format(*entries) + "\n"
    templates = []
    for entry, w in zip(entries, entry_widths):
        templates.append((entry, "{:<%dg}  " % w, " " * (w + 2)))
    return header, templates


# Copy from pytorch-pfn-extras
def filter_and_sort_entries(
    all_entries: list[str],
    unit: str = "epoch",
) -> list[str]:
    entries = deepcopy(all_entries)
    # TODO(nakago): sort other entries if necessary

    if "step" in entries:
        # move iteration to head
        entries.pop(entries.index("step"))
        if unit == "step":
            entries = ["step"] + entries
    if "epoch" in entries:
        # move epoch to head
        entries.pop(entries.index("epoch"))
        if unit == "epoch":
            entries = ["epoch"] + entries

    return entries


class PrintMetrics(Callback):
    """A callback to print the accumulated results.
    This extension uses the log accumulated by a :class:`LogReport` extension
    to print specified entries of the log in a human-readable format.
    Args:
        entries (list of str ot None): List of keys of observations to print.
            If `None` is passed, automatically infer keys from reported dict.
        log_report (str or LogReport): Log report to accumulate the
            observations. This is either the name of a LogReport extensions
            registered to the manager, or a LogReport instance to use
            internally.
        out: Stream to print the bar. Standard output is used by default.
    """

    def __init__(
        self,
        entries: list[str] | None = None,
        entry_filter: Callable | None = None,
        # out = sys.stdout,  # tqdm.write ?
        out=tqdm.tqdm,
    ) -> None:
        super().__init__()
        if entries is None:
            self._infer_entries = True
            entries = []
        else:
            self._infer_entries = False
        self._entries = entries
        self._entry_filter = entry_filter
        self._out = out

        self._log = []
        self._log_len = 0

        # format information
        header, templates = create_header_and_templates(entries)
        self._header: str | None = header
        self._templates = templates
        self._all_entries: list[str] = []

    def _update_entries(self, trainer) -> None:
        updated_flag = False
        entries = self._log[self._log_len :]
        for obs in entries:
            for entry in obs.keys():
                if entry not in self._all_entries:
                    self._all_entries.append(entry)
                    updated_flag = True

        if updated_flag:
            unit = "epoch"

            entries = filter_and_sort_entries(self._all_entries, unit=unit)
            self._entries = entries
            header, templates = create_header_and_templates(entries)
            self._header = header
            self._templates = templates

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._update_log(trainer)

        if self._infer_entries:
            self._update_entries(trainer)

        if self._header:
            self._write(self._header)
            self._header = None

        log_len = self._log_len
        while len(self._log) > log_len:
            # delete the printed contents from the current cursor
            if os.name == "nt":
                erase_console(0, 0)
            else:
                self._write("\033[J")
            self._print(self._log[log_len])
            log_len += 1
        self._log_len = log_len

    @rank_zero_only
    def _update_log(self, trainer: "pl.Trainer"):
        metrics = deepcopy(trainer.logged_metrics)

        if metrics:
            if self._entry_filter:
                entry_filter = self._entry_filter
            else:
                entry_filter = always_true  # pass all keys.

            metrics = {key: float(value) for key, value in metrics.items() if entry_filter(key)}

            metrics.update(epoch=trainer.current_epoch, step=trainer.global_step)
            self._log.append(metrics)

    @rank_zero_only
    def _print(self, observation: dict[str, float]) -> None:
        s = ""
        for entry, template, empty in self._templates:
            if entry in observation:
                s += template.format(observation[entry])
            else:
                s += empty
        self._write(s + "\n")
        self._flush()

    @rank_zero_only
    def _write(self, s):
        try:
            self._out.write(s, end="")
        except:  # noqa
            self._out.write(s)

    @rank_zero_only
    def _flush(self):
        if hasattr(self._out, "flush"):
            self._out.flush()

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


def white_filter(keywords: list[str] | str | None = None, strict: bool = False):
    if keywords is None:
        keywords = []
    if isinstance(keywords, str):
        keywords = [keywords]

    def f(v):
        for keyword in keywords:
            if strict and v == keyword:
                return True
            if not strict and keyword in v:
                return True
        return False

    return f


def black_filter(keywords: list[str] | None = None, strict: bool = False):
    white = white_filter(keywords, strict)

    def f(v):
        return not white(v)

    return f
