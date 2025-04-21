import warnings
from collection.abc import Callable
from typing import Any

import torch.nn as nn


class layerFactory:
    """
    A factory class for creating layers in a neural network.
    """
    def __init__(self, name: str, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the layer factory with the specified name and arguments.
        Args:
            name (str): The name of the layer type to create.
            *args: Positional arguments for the layer constructor.
            **kwargs: Keyword arguments for the layer constructor.
        """
        self.name = name
        self.args = args
        self.kwargs = kwargs

    def add_factory_callable():
        """
        Add the factory function to this object under the given name, with optional description. 
        """
