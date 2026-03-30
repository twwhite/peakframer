"""Shared utilities — logging setup and device detection."""

import logging

import torch
from rich.logging import RichHandler


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(show_path=False, markup=True)],
    )
    return logging.getLogger(name)


def set_log_level(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.getLogger("peakframer").setLevel(level)


def detect_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
