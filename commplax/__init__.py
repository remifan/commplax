import importlib.metadata

from .module import (
    # Core module utilities
    scan_with as scan_with,
    # Composable ensemble step primitives
    pipe as pipe,
    allreduce as allreduce,
)

__version__ = importlib.metadata.version("commplax")