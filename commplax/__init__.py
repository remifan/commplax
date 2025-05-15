import importlib.metadata

from .module import (
    scan as scan,
)

__version__ = importlib.metadata.version("commplax")