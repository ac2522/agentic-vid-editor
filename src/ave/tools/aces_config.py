"""ACES 2.0 configuration support.

Provides default working-space constants, builtin OCIO config resolution,
color-space listing, and validation helpers. PyOpenColorIO is lazily imported
and only required for functions that query an actual config.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_WORKING_SPACE = "ACES - ACEScct"
DEFAULT_OCIO_CONFIG = "ocio://cg-config-v2.2.0_aces-v2.0_no-nesting"
DEFAULT_ODT = "ACES 1.0 - SDR Video - Rec.709"


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class AcesConfigError(Exception):
    """Raised when ACES config operations fail."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_ocio():
    """Lazily import PyOpenColorIO with a clear error message."""
    try:
        import PyOpenColorIO as OCIO  # noqa: N811

        return OCIO
    except ImportError:
        raise ImportError(
            "Missing optional dependency 'PyOpenColorIO'. Install with: pip install opencolorio"
        ) from None


def _get_config(config_path: str | None):
    """Return an OCIO Config object for the given path or the builtin config."""
    OCIO = _import_ocio()
    try:
        if config_path is None:
            cfg_uri = get_builtin_aces_config()
            return OCIO.Config.CreateFromFile(cfg_uri)
        else:
            return OCIO.Config.CreateFromFile(config_path)
    except Exception as exc:
        raise AcesConfigError(
            f"Failed to load OCIO config '{config_path or cfg_uri}': {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_builtin_aces_config() -> str:
    """Return the builtin ACES config identifier string.

    For OCIO 2.5+ this is a ``ocio://`` URI that resolves to a built-in
    config. For older OCIO versions (or when OCIO is not installed) the
    same URI string is returned — validation happens at usage time.
    """
    return DEFAULT_OCIO_CONFIG


def list_colorspaces(config_path: str | None = None) -> list[str]:
    """List all colour-space names in an OCIO config.

    Args:
        config_path: Path to an OCIO config file, or *None* to use the
            builtin ACES 2.0 config.

    Returns:
        Sorted list of colour-space name strings.

    Raises:
        AcesConfigError: If the config cannot be loaded.
        ImportError: If PyOpenColorIO is not installed.
    """
    cfg = _get_config(config_path)
    return [cs.getName() for cs in cfg.getColorSpaces()]


def validate_colorspace(name: str, config_path: str | None = None) -> bool:
    """Check whether a colour-space name exists in an OCIO config.

    Args:
        name: Colour-space name to look up.
        config_path: Path to an OCIO config file, or *None* to use the
            builtin ACES 2.0 config.

    Returns:
        *True* if the name is a valid colour space, *False* otherwise.

    Raises:
        AcesConfigError: If the config itself cannot be loaded.
        ImportError: If PyOpenColorIO is not installed.
    """
    try:
        spaces = list_colorspaces(config_path)
    except AcesConfigError:
        raise
    return name in spaces
