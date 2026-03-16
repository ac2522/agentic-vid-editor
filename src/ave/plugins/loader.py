"""Plugin lazy loader — imports plugin code on first tool invocation."""

from __future__ import annotations

import importlib.util
import sys

from ave.agent.registry import ToolRegistry, RegistryError
from ave.plugins.discovery import PluginManifest


class PluginLoader:
    """Lazy-loads plugin code on first tool invocation."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry
        self._manifests: dict[str, PluginManifest] = {}
        self._loaded: set[str] = set()
        self._failed: dict[str, str] = {}  # plugin_name -> error message

    def register_manifest(
        self, manifest: PluginManifest, namespace: str = "user"
    ) -> None:
        """Register plugin summaries into the registry (no code loaded)."""
        self._manifests[manifest.name] = manifest
        for tool_stub in manifest.tools:
            self._registry.register_stub(
                name=tool_stub.name,
                domain=manifest.domain,
                summary=tool_stub.summary,
                namespace=namespace,
                plugin_name=manifest.name,
            )

    def is_loaded(self, plugin_name: str) -> bool:
        return plugin_name in self._loaded

    def call_plugin_tool(
        self, plugin_name: str, tool_name: str, params: dict
    ) -> object:
        """Load plugin if needed, then call the tool."""
        if plugin_name in self._failed:
            raise RuntimeError(
                f"Plugin '{plugin_name}' failed to load: {self._failed[plugin_name]}"
            )
        if plugin_name not in self._loaded:
            self._load_plugin(plugin_name)
        manifest = self._manifests[plugin_name]
        full_name = f"user:{manifest.domain}.{tool_name}"
        return self._registry.call_tool(full_name, params)

    def _load_plugin(self, plugin_name: str) -> None:
        """Import plugin module and call its register() function."""
        manifest = self._manifests.get(plugin_name)
        if manifest is None or manifest.path is None:
            raise RuntimeError(f"Unknown plugin: {plugin_name}")

        init_path = manifest.path / "__init__.py"
        module_name = f"ave_plugin_{manifest.name.replace('-', '_')}"

        try:
            spec = importlib.util.spec_from_file_location(
                module_name, init_path
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load {init_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            if hasattr(module, "register"):
                # Remove stubs before registering real tools
                for tool_stub in manifest.tools:
                    full_name = f"user:{manifest.domain}.{tool_stub.name}"
                    storage = self._registry._ns_to_short.get(full_name)
                    if storage and self._registry._tools.get(storage, {}).get("stub"):
                        del self._registry._tools[storage]
                        del self._registry._ns_to_short[full_name]
                        short_list = self._registry._short_names.get(
                            tool_stub.name, []
                        )
                        if full_name in short_list:
                            short_list.remove(full_name)

                module.register(self._registry, namespace="user")

            self._loaded.add(plugin_name)
        except Exception as exc:
            self._failed[plugin_name] = str(exc)
            raise RuntimeError(
                f"Plugin '{plugin_name}' failed to load: {exc}"
            ) from exc
