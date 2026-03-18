"""Proxy-first editing workflow.

Manages proxy path resolution and proxy-to-full-res conform for final render.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProxyConfig:
    """Configuration for proxy-first editing."""

    proxy_width: int = 854
    proxy_height: int = 480
    proxy_codec: str = "libx264"
    proxy_preset: str = "ultrafast"


class ConformError(Exception):
    """Raised when proxy-to-full-res conform fails."""

    def __init__(self, missing_files: list[Path]) -> None:
        self.missing_files = missing_files
        super().__init__(f"Missing full-res files: {[str(p) for p in missing_files]}")


@dataclass(frozen=True)
class ConformResult:
    """Result of a proxy-to-full-res conform operation."""

    swaps: int
    warnings: list[str]


class ProxyWorkflow:
    """Manages proxy-first editing workflow.

    - get_editing_path: returns proxy if available, else full-res
    - conform_timeline: swaps proxy refs to full-res in XGES XML
    """

    def __init__(self, config: ProxyConfig | None = None) -> None:
        self._config = config or ProxyConfig()

    @property
    def config(self) -> ProxyConfig:
        return self._config

    def get_editing_path(self, asset_entry: dict) -> Path:
        """Return proxy path for editing, full-res path if proxy unavailable.

        asset_entry has keys: 'proxy_path', 'working_path'
        """
        proxy_path = asset_entry.get("proxy_path")
        if proxy_path is not None and Path(proxy_path).exists():
            return Path(proxy_path)
        return Path(asset_entry["working_path"])

    def conform_timeline(self, xges_content: str, path_mapping: dict[str, str]) -> ConformResult:
        """Replace proxy paths with full-res paths in XGES XML content.

        path_mapping: {proxy_path: full_res_path}
        Validates all full-res paths exist BEFORE making replacements.
        Raises ConformError if any full-res paths are missing.
        Returns ConformResult with swap count.
        """
        # Validate all full-res paths exist
        missing = [
            Path(full_path) for full_path in path_mapping.values() if not Path(full_path).exists()
        ]
        if missing:
            raise ConformError(missing)

        # Perform replacements and count swaps
        total_swaps = 0
        result = xges_content
        warnings: list[str] = []
        for proxy_path, full_path in path_mapping.items():
            count = result.count(proxy_path)
            total_swaps += count
            result = result.replace(proxy_path, full_path)

        return ConformResult(swaps=total_swaps, warnings=warnings)

    def build_path_mapping(self, registry_entries: list[dict]) -> dict[str, str]:
        """Build proxy->full-res path mapping from asset registry entries.

        Each entry has 'proxy_path' and 'working_path' keys.
        """
        return {
            entry["proxy_path"]: entry["working_path"]
            for entry in registry_entries
            if entry.get("proxy_path") is not None
        }
