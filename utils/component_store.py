from __future__ import annotations

from collections import namedtuple
from collections.abc import Iterable
from keyword import iskeyword
from textwrap import dedent, indent
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def is_variable(name):
    """Returns True if `name` is a valid Python variable name and also not a keyword."""
    return name.isidentifier() and not iskeyword(name)

class ComponentStore:
    def __init__(self,name: str, description: str) -> None:
        
        self.components: dict[str, ComponentStore._Component] = {}
        self.name: str = name
        self.description: str = description

        self.__doc__ = f"Component Store '{name}': {description}\n{self.__doc__ or ''}".strip()        
        
    def add(self, name: str, desc: str, value: T) -> T:
        """Store the object `value` under the name `name` with description `desc`."""
        if not is_variable(name):
            raise ValueError("Name of component must be valid Python identifier")

        self.components[name] = self._Component(desc, value)
        return value

    def add_def(self, name: str, desc: str) -> Callable:
        """Returns a decorator which stores the decorated function under `name` with description `desc`."""

        def deco(func):
            """Decorator to add a function to a store."""
            return self.add(name, desc, func)

        return deco