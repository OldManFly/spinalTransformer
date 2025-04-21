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
    
    @property

    def names(self) -> tuple[str, ...]:
        """
        Produces all factory names.
        """
        return tuple(self.components)

    def __contains__(self, name: str) -> bool:
        """Returns True if the given name is stored."""
        return name in self.components

    def __len__(self) -> int:
        """Returns the number of stored components."""
        return len(self.components)

    def __iter__(self) -> Iterable:
        """Yields name/component pairs."""
        for k, v in self.components.items():
            yield k, v.value

    def __str__(self):
        result = f"Component Store '{self.name}': {self.description}\nAvailable components:"
        for k, v in self.components.items():
            result += f"\n* {k}:"

            if hasattr(v.value, "__doc__") and v.value.__doc__:
                doc = indent(dedent(v.value.__doc__.lstrip("\n").rstrip()), "    ")
                result += f"\n{doc}\n"
            else:
                result += f" {v.description}"

        return result

    def __getattr__(self, name: str) -> Any:
        """Returns the stored object under the given name."""
        if name in self.components:
            return self.components[name].value
        else:
            return self.__getattribute__(name)

    def __getitem__(self, name: str) -> Any:
        """Returns the stored object under the given name."""
        if name in self.components:
            return self.components[name].value
        else:
            raise ValueError(f"Component '{name}' not found")
