"""Centralized optional dependency imports.

Provides a single helper for importing optional packages with clear error
messages pointing to the correct pip extra. Follows the pandas pattern
(pandas/compat/_optional.py).
"""

from __future__ import annotations

import importlib
from types import ModuleType

# Maps package import name to the pip install extra
INSTALL_MAPPING: dict[str, str] = {
    "numpy": "vision",
    "onnxruntime": "vision",
    "transformers": "vision",
    "torch": "vision",
    "PIL": "vision",
    "scenedetect": "scene",
    "pywhispercpp": "pip install pywhispercpp",
}


def import_optional(name: str, extra: str | None = None) -> ModuleType:
    """Import an optional dependency, raising a clear error if missing.

    Args:
        name: Module name to import (e.g. "numpy", "scenedetect").
        extra: Override the install instruction. If None, looks up INSTALL_MAPPING.

    Returns:
        The imported module.

    Raises:
        ImportError: With install instructions if the module is missing.
    """
    try:
        return importlib.import_module(name)
    except ImportError:
        if extra is None:
            extra = INSTALL_MAPPING.get(name, name)
        if extra.startswith("pip install"):
            install_cmd = extra
        else:
            install_cmd = f"pip install ave[{extra}]"
        raise ImportError(
            f"Missing optional dependency '{name}'. Install with: {install_cmd}"
        ) from None
