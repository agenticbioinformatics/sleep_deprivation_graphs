"""Shared config.yaml loading and CLI-default wiring."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_config(path: str | Path | None) -> dict:
    """Load config.yaml, or return {} if path is None / doesn't exist."""
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def add_config_arg(parser: argparse.ArgumentParser, default: str = "config.yaml") -> None:
    parser.add_argument(
        "--config",
        default=default,
        help=f"Path to config.yaml with shared defaults (default: {default}).",
    )


def get(config: dict, dotted_key: str, default=None):
    """Look up 'a.b.c' in a nested config dict."""
    node = config
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
