"""Shared utilities: config loading, seeding, device selection, logging, JSON I/O."""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {config_path.resolve()}. "
            f"Pass --config pointing at your config.yaml."
        )
    with open(config_path, encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Config file at {config_path} is not valid YAML: {e}") from e
    if not isinstance(config, dict):
        raise ValueError(f"Config file at {config_path} did not parse into a dict.")
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(requested: str = "auto"):
    import torch

    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            logging.warning(
                "device='cuda' requested but no CUDA device is available. "
                "Falling back to CPU — this will be slower (spec §15)."
            )
            return torch.device("cpu")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    logging.info("No CUDA device found. Using CPU.")
    return torch.device("cpu")


def setup_logging(log_dir: str | Path, level: str = "INFO") -> None:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run.log"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
        force=True,
    )


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Expected JSON file not found: {path.resolve()}")
    with path.open("r") as f:
        return json.load(f)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, default=str)
