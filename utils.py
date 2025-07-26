"""Provides useful functions for saving and loading model and output dictionary."""

import json
from typing import Any

import torch
import torch.nn as nn


def save_model(model: nn.Module, filename: str) -> None:
    """Save model state dictionary."""
    torch.save(model.state_dict(), filename)
    print('Model saved!')


def load_model(model: nn.Module, filename: str) -> None:
    """Load model state dictionary."""
    model.load_state_dict(torch.load(filename))
    print('Model loaded!')


def save_to_json(obj: Any, filename: str) -> None:
    """Save object to .json file."""
    with open(filename, 'w') as file:
        json.dump(obj, file)


def load_from_json(filename: str) -> Any:
    """Load object from .json file."""
    with open(filename, 'r') as file:
        loaded_obj = json.load(file)
    return loaded_obj
