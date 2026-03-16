# Phase 9: Open-World Agent Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform AVE into an extensible, open-world agent platform with plugin/skill system, MCP server exposure, web search research agent, and AI rotoscoping.

**Architecture:** Four pillars implemented sequentially — P1 (Plugin/Skill System) is foundational, P2 (MCP Server) builds on P1's registry extensions, P3 (Web Search) and P4 (Rotoscoping) are exemplar plugins demonstrating the system.

**Tech Stack:** Python 3.12+, FastMCP 2.x, Brave Search API, SAM 2 (PyTorch), Robust Video Matting, Pydantic, pytest

**Spec:** `docs/superpowers/specs/2026-03-16-phase9-open-world-agent-design.md`

---

## Chunk 1: Plugin/Skill System — Core Infrastructure

### File Structure

```
src/ave/plugins/__init__.py          # Plugin loader, PluginManifest, PluginRegistry
src/ave/plugins/discovery.py         # Directory scanning, manifest parsing
src/ave/skills/__init__.py           # Skill loader, SkillMeta, SkillRegistry
src/ave/skills/discovery.py          # Directory scanning, frontmatter parsing
src/ave/agent/registry.py            # MODIFY: add namespace support
src/ave/agent/session.py             # MODIFY: integrate plugin/skill loading
tests/test_plugins/__init__.py
tests/test_plugins/test_discovery.py
tests/test_plugins/test_loader.py
tests/test_plugins/test_namespace.py
tests/test_skills/__init__.py
tests/test_skills/test_discovery.py
tests/test_skills/test_loader.py
```

### Task 1: Plugin Manifest Data Model

**Files:**
- Create: `src/ave/plugins/__init__.py`
- Test: `tests/test_plugins/__init__.py`, `tests/test_plugins/test_discovery.py`

- [ ] **Step 1: Write failing test for PluginManifest parsing**

```python
# tests/test_plugins/test_discovery.py
from __future__ import annotations

import pytest
from ave.plugins.discovery import PluginManifest, parse_manifest


class TestPluginManifest:
    def test_parse_valid_manifest(self, tmp_path):
        manifest = tmp_path / "plugin.yaml"
        manifest.write_text(
            "name: test-plugin\n"
            "description: A test plugin\n"
            "version: 1.0.0\n"
            "domain: editing\n"
            "tools:\n"
            "  - name: my_tool\n"
            "    summary: Does something\n"
        )
        result = parse_manifest(manifest)
        assert result.name == "test-plugin"
        assert result.description == "A test plugin"
        assert result.version == "1.0.0"
        assert result.domain == "editing"
        assert len(result.tools) == 1
        assert result.tools[0].name == "my_tool"
        assert result.tools[0].summary == "Does something"

    def test_parse_manifest_with_requirements(self, tmp_path):
        manifest = tmp_path / "plugin.yaml"
        manifest.write_text(
            "name: gpu-plugin\n"
            "description: Needs GPU\n"
            "version: 0.1.0\n"
            "domain: vfx\n"
            "tools: []\n"
            "requires:\n"
            "  python: [torch]\n"
            "  system: [cuda]\n"
        )
        result = parse_manifest(manifest)
        assert result.requires_python == ["torch"]
        assert result.requires_system == ["cuda"]

    def test_parse_manifest_missing_required_field(self, tmp_path):
        manifest = tmp_path / "plugin.yaml"
        manifest.write_text("name: incomplete\n")
        with pytest.raises(ValueError, match="description"):
            parse_manifest(manifest)

    def test_manifest_is_frozen(self, tmp_path):
        manifest = tmp_path / "plugin.yaml"
        manifest.write_text(
            "name: frozen\n"
            "description: test\n"
            "version: 1.0.0\n"
            "domain: test\n"
            "tools: []\n"
        )
        result = parse_manifest(manifest)
        with pytest.raises(AttributeError):
            result.name = "changed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_plugins/test_discovery.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ave.plugins'`

- [ ] **Step 3: Implement PluginManifest and parse_manifest**

```python
# src/ave/plugins/__init__.py
"""Plugin system for AVE — lazy-loaded Python capability extensions."""
from __future__ import annotations

# src/ave/plugins/discovery.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ToolStub:
    """Minimal tool info from plugin manifest (not the full implementation)."""
    name: str
    summary: str


@dataclass(frozen=True)
class PluginManifest:
    """Parsed plugin.yaml — loaded at startup, code loaded on demand."""
    name: str
    description: str
    version: str
    domain: str
    tools: tuple[ToolStub, ...]
    requires_python: tuple[str, ...] = ()
    requires_system: tuple[str, ...] = ()
    path: Path | None = None


_REQUIRED_FIELDS = ("name", "description", "version", "domain")


def parse_manifest(manifest_path: Path) -> PluginManifest:
    """Parse a plugin.yaml file into a PluginManifest."""
    with open(manifest_path) as f:
        data: dict[str, Any] = yaml.safe_load(f)

    for field_name in _REQUIRED_FIELDS:
        if field_name not in data:
            raise ValueError(
                f"plugin.yaml missing required field: {field_name}"
            )

    tools_raw = data.get("tools", [])
    tools = tuple(
        ToolStub(name=t["name"], summary=t.get("summary", ""))
        for t in tools_raw
    )

    requires = data.get("requires", {})
    return PluginManifest(
        name=data["name"],
        description=data["description"],
        version=data["version"],
        domain=data["domain"],
        tools=tools,
        requires_python=tuple(requires.get("python", [])),
        requires_system=tuple(requires.get("system", [])),
        path=manifest_path.parent,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_plugins/test_discovery.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ave/plugins/ tests/test_plugins/
git commit -m "feat(plugins): add PluginManifest data model and parser"
```

---

### Task 2: Plugin Directory Discovery

**Files:**
- Modify: `src/ave/plugins/discovery.py`
- Test: `tests/test_plugins/test_discovery.py`

- [ ] **Step 1: Write failing test for discover_plugins**

```python
# tests/test_plugins/test_discovery.py (append)

class TestDiscoverPlugins:
    def test_discover_from_single_directory(self, tmp_path):
        plugin_dir = tmp_path / "my-plugin"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.yaml").write_text(
            "name: my-plugin\n"
            "description: test\n"
            "version: 1.0.0\n"
            "domain: editing\n"
            "tools:\n"
            "  - name: do_thing\n"
            "    summary: Does a thing\n"
        )
        from ave.plugins.discovery import discover_plugins

        manifests = discover_plugins([tmp_path])
        assert len(manifests) == 1
        assert manifests[0].name == "my-plugin"

    def test_discover_respects_priority_order(self, tmp_path):
        """Earlier directories have higher priority — last wins on conflict."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        for d in (dir_a, dir_b):
            plugin = d / "same-name"
            plugin.mkdir(parents=True)
            (plugin / "plugin.yaml").write_text(
                f"name: same-name\n"
                f"description: from {d.name}\n"
                f"version: 1.0.0\n"
                f"domain: editing\n"
                f"tools: []\n"
            )
        manifests = discover_plugins([dir_a, dir_b])
        assert len(manifests) == 1
        # dir_a is higher priority (project > user > builtin)
        assert manifests[0].description == "from a"

    def test_discover_skips_directories_without_manifest(self, tmp_path):
        no_manifest = tmp_path / "empty-plugin"
        no_manifest.mkdir()
        manifests = discover_plugins([tmp_path])
        assert len(manifests) == 0

    def test_discover_skips_nonexistent_directories(self):
        manifests = discover_plugins([Path("/nonexistent/path")])
        assert len(manifests) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_plugins/test_discovery.py::TestDiscoverPlugins -v`
Expected: FAIL — `ImportError: cannot import name 'discover_plugins'`

- [ ] **Step 3: Implement discover_plugins**

```python
# src/ave/plugins/discovery.py (append)

def discover_plugins(search_dirs: list[Path]) -> list[PluginManifest]:
    """Scan directories for plugin.yaml files. Earlier dirs = higher priority."""
    seen_names: dict[str, PluginManifest] = {}

    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
        for child in sorted(search_dir.iterdir()):
            if not child.is_dir():
                continue
            manifest_path = child / "plugin.yaml"
            if not manifest_path.exists():
                continue
            try:
                manifest = parse_manifest(manifest_path)
            except (ValueError, yaml.YAMLError):
                continue
            if manifest.name not in seen_names:
                seen_names[manifest.name] = manifest

    return list(seen_names.values())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_plugins/test_discovery.py -v`
Expected: PASS (all 8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ave/plugins/discovery.py tests/test_plugins/test_discovery.py
git commit -m "feat(plugins): add directory discovery with priority ordering"
```

---

### Task 3: Skill Frontmatter Data Model and Discovery

**Files:**
- Create: `src/ave/skills/__init__.py`, `src/ave/skills/discovery.py`
- Test: `tests/test_skills/__init__.py`, `tests/test_skills/test_discovery.py`

- [ ] **Step 1: Write failing test for SkillMeta parsing and discovery**

```python
# tests/test_skills/test_discovery.py
from __future__ import annotations

import pytest
from ave.skills.discovery import SkillMeta, parse_skill_frontmatter, discover_skills
from pathlib import Path


class TestSkillMeta:
    def test_parse_frontmatter(self, tmp_path):
        skill = tmp_path / "grain.md"
        skill.write_text(
            "---\n"
            "name: cinematic-grain\n"
            "description: Apply cinematic film grain\n"
            "domain: color\n"
            "triggers:\n"
            "  - film grain\n"
            "  - grain effect\n"
            "---\n"
            "\n## Steps\n1. Do the thing\n"
        )
        meta = parse_skill_frontmatter(skill)
        assert meta.name == "cinematic-grain"
        assert meta.description == "Apply cinematic film grain"
        assert meta.domain == "color"
        assert meta.triggers == ("film grain", "grain effect")
        assert meta.path == skill

    def test_parse_missing_frontmatter(self, tmp_path):
        skill = tmp_path / "bad.md"
        skill.write_text("# No frontmatter here\n")
        with pytest.raises(ValueError, match="frontmatter"):
            parse_skill_frontmatter(skill)


class TestDiscoverSkills:
    def test_discover_skills_from_directory(self, tmp_path):
        (tmp_path / "skill1.md").write_text(
            "---\nname: s1\ndescription: first\ndomain: editing\ntriggers: []\n---\nbody\n"
        )
        (tmp_path / "skill2.md").write_text(
            "---\nname: s2\ndescription: second\ndomain: color\ntriggers: []\n---\nbody\n"
        )
        (tmp_path / "not-a-skill.txt").write_text("ignored")
        skills = discover_skills([tmp_path])
        assert len(skills) == 2
        names = {s.name for s in skills}
        assert names == {"s1", "s2"}

    def test_discover_priority_order(self, tmp_path):
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        for d in (dir_a, dir_b):
            (d / "same.md").write_text(
                f"---\nname: same\ndescription: from {d.name}\n"
                f"domain: editing\ntriggers: []\n---\nbody\n"
            )
        skills = discover_skills([dir_a, dir_b])
        assert len(skills) == 1
        assert skills[0].description == "from a"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_skills/test_discovery.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ave.skills'`

- [ ] **Step 3: Implement SkillMeta, parse_skill_frontmatter, discover_skills**

```python
# src/ave/skills/__init__.py
"""Skill system for AVE — markdown workflow playbooks loaded on demand."""
from __future__ import annotations

# src/ave/skills/discovery.py
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass(frozen=True)
class SkillMeta:
    """Frontmatter metadata — loaded at startup. Body loaded on demand."""
    name: str
    description: str
    domain: str
    triggers: tuple[str, ...]
    path: Path


def parse_skill_frontmatter(skill_path: Path) -> SkillMeta:
    """Parse YAML frontmatter from a skill markdown file."""
    text = skill_path.read_text()
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise ValueError(f"No YAML frontmatter found in {skill_path}")

    data: dict[str, Any] = yaml.safe_load(match.group(1))
    return SkillMeta(
        name=data["name"],
        description=data["description"],
        domain=data.get("domain", "general"),
        triggers=tuple(data.get("triggers", [])),
        path=skill_path,
    )


def discover_skills(search_dirs: list[Path]) -> list[SkillMeta]:
    """Scan directories for .md skill files. Earlier dirs = higher priority."""
    seen_names: dict[str, SkillMeta] = {}

    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
        for md_file in sorted(search_dir.glob("*.md")):
            try:
                meta = parse_skill_frontmatter(md_file)
            except (ValueError, yaml.YAMLError, KeyError):
                continue
            if meta.name not in seen_names:
                seen_names[meta.name] = meta

    return list(seen_names.values())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_skills/test_discovery.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ave/skills/ tests/test_skills/
git commit -m "feat(skills): add SkillMeta frontmatter parsing and discovery"
```

---

### Task 4: Domain Namespacing in ToolRegistry

**Files:**
- Modify: `src/ave/agent/registry.py`
- Test: `tests/test_plugins/test_namespace.py`

- [ ] **Step 1: Write failing test for namespaced tool registration and search**

```python
# tests/test_plugins/test_namespace.py
from __future__ import annotations

import pytest
from ave.agent.registry import ToolRegistry


class TestNamespacedTools:
    def test_register_tool_with_namespace(self):
        reg = ToolRegistry()

        @reg.tool(domain="editing", namespace="ave")
        def trim(duration_ns: int) -> dict:
            """Trim a clip."""
            return {"trimmed": duration_ns}

        schema = reg.get_tool_schema("ave:editing.trim")
        assert schema.name == "ave:editing.trim"
        assert schema.domain == "editing"

    def test_default_namespace_is_ave(self):
        reg = ToolRegistry()

        @reg.tool(domain="color")
        def grade(intensity: float) -> dict:
            """Color grade."""
            return {"graded": intensity}

        schema = reg.get_tool_schema("ave:color.grade")
        assert schema.name == "ave:color.grade"

    def test_search_across_namespaces(self):
        reg = ToolRegistry()

        @reg.tool(domain="editing", namespace="ave")
        def trim(duration_ns: int) -> dict:
            """Trim a clip."""
            return {}

        @reg.tool(domain="editing", namespace="user")
        def smart_trim(duration_ns: int) -> dict:
            """AI-powered trim."""
            return {}

        results = reg.search_tools("trim")
        names = [r.name for r in results]
        assert "ave:editing.trim" in names
        assert "user:editing.smart_trim" in names

    def test_call_tool_by_namespaced_name(self):
        reg = ToolRegistry()

        @reg.tool(domain="editing", namespace="user")
        def my_tool(x: int) -> dict:
            """Custom tool."""
            return {"result": x * 2}

        result = reg.call_tool("user:editing.my_tool", {"x": 5})
        assert result == {"result": 10}

    def test_backward_compat_unnamespaced_call(self):
        """Existing code calling tools by short name still works."""
        reg = ToolRegistry()

        @reg.tool(domain="editing")
        def trim(duration_ns: int) -> dict:
            """Trim."""
            return {"trimmed": duration_ns}

        # Short name lookup falls back when unambiguous
        result = reg.call_tool("trim", {"duration_ns": 100})
        assert result == {"trimmed": 100}

    def test_ambiguous_short_name_raises(self):
        reg = ToolRegistry()

        @reg.tool(domain="editing", namespace="ave")
        def trim(x: int) -> dict:
            """Trim A."""
            return {}

        @reg.tool(domain="editing", namespace="user")
        def trim(x: int) -> dict:  # noqa: F811
            """Trim B."""
            return {}

        with pytest.raises(KeyError, match="ambiguous"):
            reg.call_tool("trim", {"x": 1})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_plugins/test_namespace.py -v`
Expected: FAIL — `TypeError: tool() got an unexpected keyword argument 'namespace'`

- [ ] **Step 3: Modify ToolRegistry to support namespaces**

Modify `src/ave/agent/registry.py`:
- Add `namespace: str = "ave"` parameter to `tool()` decorator
- Store tools with namespaced key: `f"{namespace}:{domain}.{func_name}"`
- Add short-name alias index: `self._short_names: dict[str, list[str]]` mapping `func_name` → list of full names
- In `call_tool`, `get_tool_schema`: try full name first, then short name lookup (raise on ambiguous)
- In `search_tools`: include namespace in `ToolSummary` — add optional `namespace` field to `ToolSummary`

- [ ] **Step 4: Run ALL tests to verify nothing breaks**

Run: `python -m pytest tests/test_plugins/test_namespace.py tests/test_agent/test_registry.py -v`
Expected: PASS — new tests pass, existing registry tests may need minor updates for namespaced names

- [ ] **Step 5: Fix any broken existing tests**

Existing tests use short names like `"trim"`. The backward-compat fallback should handle this, but if any tests check exact names in `ToolSummary.name`, update them to expect `"ave:editing.trim"` format.

- [ ] **Step 6: Commit**

```bash
git add src/ave/agent/registry.py tests/test_plugins/test_namespace.py tests/test_agent/test_registry.py
git commit -m "feat(registry): add namespace support with backward-compatible short names"
```

---

### Task 5: Plugin Lazy Loader

**Files:**
- Create: `src/ave/plugins/loader.py`
- Test: `tests/test_plugins/test_loader.py`

- [ ] **Step 1: Write failing test for lazy plugin loading**

```python
# tests/test_plugins/test_loader.py
from __future__ import annotations

import pytest
from pathlib import Path
from ave.plugins.loader import PluginLoader
from ave.plugins.discovery import PluginManifest, ToolStub
from ave.agent.registry import ToolRegistry


def _make_plugin(tmp_path: Path, name: str = "test-plugin") -> PluginManifest:
    """Create a real plugin directory with register function."""
    plugin_dir = tmp_path / name
    plugin_dir.mkdir()
    (plugin_dir / "plugin.yaml").write_text(
        f"name: {name}\n"
        f"description: Test\n"
        f"version: 1.0.0\n"
        f"domain: testing\n"
        f"tools:\n"
        f"  - name: greet\n"
        f"    summary: Says hello\n"
    )
    (plugin_dir / "__init__.py").write_text(
        "def register(registry, namespace):\n"
        "    @registry.tool(domain='testing', namespace=namespace)\n"
        "    def greet(name: str) -> dict:\n"
        "        '''Say hello.'''\n"
        "        return {'greeting': f'hello {name}'}\n"
    )
    return PluginManifest(
        name=name,
        description="Test",
        version="1.0.0",
        domain="testing",
        tools=(ToolStub(name="greet", summary="Says hello"),),
        path=plugin_dir,
    )


class TestPluginLoader:
    def test_register_summaries_without_loading_code(self, tmp_path):
        manifest = _make_plugin(tmp_path)
        registry = ToolRegistry()
        loader = PluginLoader(registry)
        loader.register_manifest(manifest)

        # Summary is searchable
        results = registry.search_tools("greet")
        assert len(results) >= 1
        # But code is NOT loaded yet
        assert not loader.is_loaded(manifest.name)

    def test_lazy_load_on_first_call(self, tmp_path):
        manifest = _make_plugin(tmp_path)
        registry = ToolRegistry()
        loader = PluginLoader(registry)
        loader.register_manifest(manifest)

        # First call triggers code loading
        result = loader.call_plugin_tool(
            manifest.name, "greet", {"name": "world"}
        )
        assert result == {"greeting": "hello world"}
        assert loader.is_loaded(manifest.name)

    def test_load_failure_marks_unavailable(self, tmp_path):
        plugin_dir = tmp_path / "broken"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("raise ImportError('missing dep')")
        manifest = PluginManifest(
            name="broken",
            description="Broken plugin",
            version="1.0.0",
            domain="testing",
            tools=(ToolStub(name="nope", summary="Won't work"),),
            path=plugin_dir,
        )
        registry = ToolRegistry()
        loader = PluginLoader(registry)
        loader.register_manifest(manifest)

        with pytest.raises(RuntimeError, match="failed to load"):
            loader.call_plugin_tool("broken", "nope", {})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_plugins/test_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ave.plugins.loader'`

- [ ] **Step 3: Implement PluginLoader**

```python
# src/ave/plugins/loader.py
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from ave.agent.registry import ToolRegistry
from ave.plugins.discovery import PluginManifest


class PluginLoader:
    """Lazy-loads plugin code on first tool invocation."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry
        self._manifests: dict[str, PluginManifest] = {}
        self._loaded: set[str] = set()
        self._failed: set[str] = set()

    def register_manifest(self, manifest: PluginManifest) -> None:
        """Register plugin summaries into the registry (no code loaded)."""
        self._manifests[manifest.name] = manifest
        namespace = f"user"  # TODO: determine from source dir
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
                f"Plugin '{plugin_name}' failed to load previously"
            )
        if plugin_name not in self._loaded:
            self._load_plugin(plugin_name)
        # After loading, tool is registered — delegate to registry
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
                module.register(self._registry, namespace="user")

            self._loaded.add(plugin_name)
        except Exception as exc:
            self._failed.add(plugin_name)
            raise RuntimeError(
                f"Plugin '{plugin_name}' failed to load: {exc}"
            ) from exc
```

Note: This requires adding a `register_stub` method to `ToolRegistry` that registers a placeholder tool entry (searchable but not callable until the real implementation is registered by the plugin's `register()` function).

- [ ] **Step 4: Add register_stub to ToolRegistry**

```python
# In src/ave/agent/registry.py, add method:
def register_stub(
    self,
    name: str,
    domain: str,
    summary: str,
    namespace: str = "user",
    plugin_name: str | None = None,
) -> None:
    """Register a tool stub (searchable, not yet callable)."""
    full_name = f"{namespace}:{domain}.{name}"
    self._tools[full_name] = {
        "func": None,  # Not loaded yet
        "domain": domain,
        "requires": [],
        "provides": [],
        "tags": [],
        "modifies_timeline": False,
        "stub": True,
        "plugin_name": plugin_name,
        "description": summary,
    }
    self._short_names.setdefault(name, []).append(full_name)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_plugins/ -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/ave/plugins/loader.py src/ave/agent/registry.py tests/test_plugins/test_loader.py
git commit -m "feat(plugins): add lazy PluginLoader with stub registration"
```

---

### Task 6: Skill Loader (On-Demand Body Loading)

**Files:**
- Create: `src/ave/skills/loader.py`
- Test: `tests/test_skills/test_loader.py`

- [ ] **Step 1: Write failing test for skill body loading**

```python
# tests/test_skills/test_loader.py
from __future__ import annotations

from pathlib import Path
from ave.skills.loader import SkillLoader
from ave.skills.discovery import SkillMeta


class TestSkillLoader:
    def test_load_skill_body(self, tmp_path):
        skill_path = tmp_path / "grain.md"
        skill_path.write_text(
            "---\n"
            "name: grain\n"
            "description: Film grain\n"
            "domain: color\n"
            "triggers: [film grain]\n"
            "---\n"
            "\n## Steps\n1. Apply grain overlay\n2. Adjust blend\n"
        )
        meta = SkillMeta(
            name="grain",
            description="Film grain",
            domain="color",
            triggers=("film grain",),
            path=skill_path,
        )
        loader = SkillLoader()
        body = loader.load_body(meta)
        assert "## Steps" in body
        assert "Apply grain overlay" in body
        assert "---" not in body  # Frontmatter stripped

    def test_match_skill_by_trigger(self, tmp_path):
        skill_path = tmp_path / "grain.md"
        skill_path.write_text(
            "---\nname: grain\ndescription: Film grain\n"
            "domain: color\ntriggers: [film grain, grain effect, analog]\n---\nbody\n"
        )
        meta = SkillMeta(
            name="grain", description="Film grain", domain="color",
            triggers=("film grain", "grain effect", "analog"),
            path=skill_path,
        )
        loader = SkillLoader()
        loader.register(meta)

        matches = loader.match("I want to add a film grain effect")
        assert len(matches) >= 1
        assert matches[0].name == "grain"

    def test_no_match_returns_empty(self, tmp_path):
        loader = SkillLoader()
        meta = SkillMeta(
            name="grain", description="Film grain", domain="color",
            triggers=("film grain",), path=tmp_path / "grain.md",
        )
        loader.register(meta)
        matches = loader.match("trim the clip at 5 seconds")
        assert len(matches) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_skills/test_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ave.skills.loader'`

- [ ] **Step 3: Implement SkillLoader**

```python
# src/ave/skills/loader.py
from __future__ import annotations

import re
from ave.skills.discovery import SkillMeta

_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


class SkillLoader:
    """Registers skill metadata, matches intents, loads body on demand."""

    def __init__(self) -> None:
        self._skills: dict[str, SkillMeta] = {}

    def register(self, meta: SkillMeta) -> None:
        self._skills[meta.name] = meta

    def load_body(self, meta: SkillMeta) -> str:
        """Load and return the skill body (frontmatter stripped)."""
        text = meta.path.read_text()
        return _FRONTMATTER_RE.sub("", text).strip()

    def match(self, intent: str, limit: int = 5) -> list[SkillMeta]:
        """Match an intent string against skill triggers and descriptions."""
        intent_lower = intent.lower()
        scored: list[tuple[float, SkillMeta]] = []

        for meta in self._skills.values():
            score = 0.0
            for trigger in meta.triggers:
                if trigger.lower() in intent_lower:
                    score += 2.0
            # Also check description words
            for word in meta.description.lower().split():
                if len(word) > 3 and word in intent_lower:
                    score += 0.5
            if score > 0:
                scored.append((score, meta))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [meta for _, meta in scored[:limit]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_skills/test_loader.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ave/skills/loader.py tests/test_skills/test_loader.py
git commit -m "feat(skills): add SkillLoader with trigger matching and body loading"
```

---

### Task 7: Integrate Plugins and Skills into EditingSession

**Files:**
- Modify: `src/ave/agent/session.py`
- Test: `tests/test_agent/test_session.py` (add new tests)

- [ ] **Step 1: Write failing test for session with plugins and skills**

```python
# tests/test_agent/test_session_plugins.py
from __future__ import annotations

import pytest
from pathlib import Path
from ave.agent.session import EditingSession


class TestSessionPluginIntegration:
    def test_session_discovers_plugins(self, tmp_path):
        plugin_dir = tmp_path / "plugins" / "hello"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(
            "name: hello\n"
            "description: Hello plugin\n"
            "version: 1.0.0\n"
            "domain: testing\n"
            "tools:\n"
            "  - name: say_hello\n"
            "    summary: Says hello\n"
        )
        (plugin_dir / "__init__.py").write_text(
            "def register(registry, namespace):\n"
            "    @registry.tool(domain='testing', namespace=namespace)\n"
            "    def say_hello(name: str) -> dict:\n"
            "        '''Say hello.'''\n"
            "        return {'msg': f'hello {name}'}\n"
        )
        session = EditingSession(
            plugin_dirs=[tmp_path / "plugins"],
        )
        results = session.search_tools("hello")
        assert any("say_hello" in r.name for r in results)

    def test_session_discovers_skills(self, tmp_path):
        skill_dir = tmp_path / "skills"
        skill_dir.mkdir()
        (skill_dir / "grain.md").write_text(
            "---\nname: grain\ndescription: Film grain effect\n"
            "domain: color\ntriggers: [film grain]\n---\n## Steps\n1. Do it\n"
        )
        session = EditingSession(
            skill_dirs=[skill_dir],
        )
        matches = session.match_skills("add film grain")
        assert len(matches) >= 1
        assert matches[0].name == "grain"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent/test_session_plugins.py -v`
Expected: FAIL — `TypeError: EditingSession.__init__() got an unexpected keyword argument 'plugin_dirs'`

- [ ] **Step 3: Modify EditingSession to accept plugin_dirs and skill_dirs**

Add to `EditingSession.__init__`:
```python
def __init__(
    self,
    snapshot_manager: SnapshotManager | None = None,
    transition_graph: ToolTransitionGraph | None = None,
    plugin_dirs: list[Path] | None = None,
    skill_dirs: list[Path] | None = None,
) -> None:
    # ... existing init ...
    self._plugin_loader: PluginLoader | None = None
    self._skill_loader = SkillLoader()

    if plugin_dirs:
        from ave.plugins.discovery import discover_plugins
        from ave.plugins.loader import PluginLoader
        self._plugin_loader = PluginLoader(self._registry)
        for manifest in discover_plugins(plugin_dirs):
            self._plugin_loader.register_manifest(manifest)

    if skill_dirs:
        from ave.skills.discovery import discover_skills
        for meta in discover_skills(skill_dirs):
            self._skill_loader.register(meta)
```

Add method:
```python
def match_skills(self, intent: str) -> list[SkillMeta]:
    """Match user intent against registered skills."""
    return self._skill_loader.match(intent)

def load_skill(self, skill_name: str) -> str:
    """Load full skill body by name."""
    meta = self._skill_loader._skills.get(skill_name)
    if meta is None:
        raise KeyError(f"Unknown skill: {skill_name}")
    return self._skill_loader.load_body(meta)
```

- [ ] **Step 4: Run all session tests**

Run: `python -m pytest tests/test_agent/test_session_plugins.py tests/test_agent/test_session.py -v`
Expected: PASS — new tests pass, existing tests unaffected (plugin_dirs/skill_dirs default to None)

- [ ] **Step 5: Commit**

```bash
git add src/ave/agent/session.py tests/test_agent/test_session_plugins.py
git commit -m "feat(session): integrate plugin and skill discovery into EditingSession"
```

---

## Chunk 2: MCP Server Exposure

### File Structure

```
src/ave/mcp/__init__.py              # MCP server entry point
src/ave/mcp/server.py                # FastMCP server with 6 tools
src/ave/mcp/types.py                 # EditResult, ProjectState, etc.
src/ave/cli.py                       # MODIFY: add 'ave mcp serve' command
tests/test_mcp/__init__.py
tests/test_mcp/test_server.py
tests/test_mcp/test_types.py
tests/test_mcp/test_e2e.py           # @pytest.mark.llm end-to-end
```

### Task 8: MCP Response Data Models

**Files:**
- Create: `src/ave/mcp/__init__.py`, `src/ave/mcp/types.py`
- Test: `tests/test_mcp/__init__.py`, `tests/test_mcp/test_types.py`

- [ ] **Step 1: Write failing test for MCP data models**

```python
# tests/test_mcp/test_types.py
from __future__ import annotations

from ave.mcp.types import (
    EditResult,
    ProjectState,
    PreviewResult,
    AssetInfo,
)


class TestEditResult:
    def test_create_success(self):
        r = EditResult(
            success=True,
            description="Added cross dissolve between clips 3 and 4",
            tools_used=["ave:editing.transition"],
            preview_path="/tmp/preview.jpg",
        )
        assert r.success is True
        assert "cross dissolve" in r.description

    def test_create_failure(self):
        r = EditResult(
            success=False,
            description="No clips found matching description",
            tools_used=[],
            error="clip_not_found",
        )
        assert r.success is False
        assert r.error == "clip_not_found"


class TestProjectState:
    def test_create_with_clips(self):
        s = ProjectState(
            clip_count=3,
            duration_ns=5_000_000_000,
            layers=1,
            clips=[{"name": "clip1", "start_ns": 0, "duration_ns": 2_000_000_000}],
        )
        assert s.clip_count == 3


class TestPreviewResult:
    def test_create(self):
        r = PreviewResult(path="/tmp/frame.jpg", format="jpeg", width=1920, height=1080)
        assert r.path == "/tmp/frame.jpg"


class TestAssetInfo:
    def test_create(self):
        a = AssetInfo(
            asset_id="abc123",
            path="/media/clip.mov",
            codec="prores",
            width=1920,
            height=1080,
            duration_ns=10_000_000_000,
            color_space="bt709",
        )
        assert a.codec == "prores"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mcp/test_types.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement data models**

```python
# src/ave/mcp/__init__.py
"""MCP server for AVE — exposes video editing to external agents."""
from __future__ import annotations

# src/ave/mcp/types.py
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EditResult:
    success: bool
    description: str
    tools_used: list[str] = field(default_factory=list)
    preview_path: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class ProjectState:
    clip_count: int
    duration_ns: int
    layers: int
    clips: list[dict] = field(default_factory=list)
    effects: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PreviewResult:
    path: str
    format: str
    width: int = 0
    height: int = 0


@dataclass(frozen=True)
class AssetInfo:
    asset_id: str
    path: str
    codec: str
    width: int
    height: int
    duration_ns: int
    color_space: str | None = None
    frame_rate: float | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_mcp/test_types.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ave/mcp/ tests/test_mcp/
git commit -m "feat(mcp): add MCP response data models"
```

---

### Task 9: MCP Server with 6 Tools

**Files:**
- Create: `src/ave/mcp/server.py`
- Test: `tests/test_mcp/test_server.py`

- [ ] **Step 1: Write failing test for MCP server tool registration**

```python
# tests/test_mcp/test_server.py
from __future__ import annotations

import pytest
from ave.mcp.server import create_mcp_server


class TestMcpServer:
    def test_server_has_six_tools(self):
        server = create_mcp_server()
        tool_names = {t.name for t in server.list_tools()}
        expected = {
            "edit_video",
            "get_project_state",
            "render_preview",
            "ingest_asset",
            "search_tools",
            "call_tool",
        }
        assert tool_names == expected

    def test_search_tools_delegates_to_registry(self):
        server = create_mcp_server()
        # search_tools should work even without a loaded project
        result = server.call_tool("search_tools", {"query": "trim"})
        assert isinstance(result, list)

    def test_get_project_state_without_project(self):
        server = create_mcp_server()
        result = server.call_tool("get_project_state", {})
        assert result["clip_count"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mcp/test_server.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement create_mcp_server**

```python
# src/ave/mcp/server.py
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ave.agent.session import EditingSession
from ave.mcp.types import EditResult, ProjectState, PreviewResult, AssetInfo

try:
    from fastmcp import FastMCP
except ImportError:
    FastMCP = None


def create_mcp_server(
    session: EditingSession | None = None,
) -> FastMCP:
    """Create AVE's MCP server with 6 outcome-oriented tools."""
    if FastMCP is None:
        raise ImportError("fastmcp is required: pip install fastmcp")

    if session is None:
        session = EditingSession()

    mcp = FastMCP("ave", description="Agentic Video Editor")

    @mcp.tool()
    def edit_video(instruction: str, options: dict | None = None) -> dict:
        """Natural language video editing. AVE's internal orchestrator
        handles tool discovery, role-based routing, execution, and
        verification.

        Examples:
          edit_video(instruction="remove all filler words")
          edit_video(instruction="add cross dissolve between clips 3 and 4")
        """
        # TODO: Wire to orchestrator agentic loop in P3
        return asdict(EditResult(
            success=False,
            description="Orchestrator not yet connected",
            error="not_implemented",
        ))

    @mcp.tool()
    def get_project_state(include: list[str] | None = None) -> dict:
        """Current timeline structure, clips, effects, metadata."""
        state = session.to_dict()
        return asdict(ProjectState(
            clip_count=len(state.get("history", [])),
            duration_ns=0,
            layers=0,
            clips=state.get("provisions", []),
        ))

    @mcp.tool()
    def render_preview(
        segment: str | None = None, format: str = "jpeg"
    ) -> dict:
        """Render a preview frame or segment. Returns file path."""
        return asdict(PreviewResult(path="", format=format))

    @mcp.tool()
    def ingest_asset(path: str, options: dict | None = None) -> dict:
        """Bring media into the project. Auto-probes codec, resolution."""
        return asdict(AssetInfo(
            asset_id="", path=path, codec="unknown",
            width=0, height=0, duration_ns=0,
        ))

    @mcp.tool()
    def search_tools(query: str, domain: str | None = None) -> list[dict]:
        """Discover AVE's granular tools by keyword or domain."""
        results = session.search_tools(query, domain)
        return [
            {"name": r.name, "domain": r.domain, "description": r.description}
            for r in results
        ]

    @mcp.tool()
    def call_tool(name: str, params: dict) -> Any:
        """Execute any registered AVE tool directly by name."""
        return session.call_tool(name, params)

    return mcp
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_mcp/test_server.py -v`
Expected: PASS (may need to install fastmcp: `pip install fastmcp`)

- [ ] **Step 5: Commit**

```bash
git add src/ave/mcp/server.py tests/test_mcp/test_server.py
git commit -m "feat(mcp): add FastMCP server with 6 outcome-oriented tools"
```

---

### Task 10: CLI Entry Point for MCP Server

**Files:**
- Modify: `src/ave/cli.py` (or create if doesn't exist)
- Test: `tests/test_mcp/test_cli.py`

- [ ] **Step 1: Write failing test for CLI**

```python
# tests/test_mcp/test_cli.py
from __future__ import annotations

import subprocess
import sys


class TestMcpCli:
    def test_mcp_serve_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "ave", "mcp", "serve", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "serve" in result.stdout.lower() or "transport" in result.stdout.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mcp/test_cli.py -v`
Expected: FAIL

- [ ] **Step 3: Implement CLI entry point**

Check if `src/ave/cli.py` or `src/ave/__main__.py` exists. Add `mcp serve` subcommand using argparse or click (follow existing CLI pattern). The command should:
- Accept `--transport` (stdio or http, default stdio)
- Accept `--port` (for http, default 8420)
- Accept `--project` (path to .xges file)
- Create session, create MCP server, run it

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_mcp/test_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ave/cli.py tests/test_mcp/test_cli.py
git commit -m "feat(cli): add 'ave mcp serve' command"
```

---

## Chunk 3: Web Search + Research Agent

### File Structure

```
src/ave/tools/search.py              # SearchBackend protocol, SearchResult, PageContent
src/ave/tools/search_brave.py        # BraveSearchBackend implementation
src/ave/agent/researcher.py          # ResearchBrief, Approach, researcher subagent logic
src/ave/agent/tools/research.py      # Register research domain tools
src/ave/agent/roles.py               # MODIFY: add RESEARCHER_ROLE
tests/test_tools/test_search.py
tests/test_tools/test_search_brave.py
tests/test_agent/test_researcher.py
tests/test_agent/test_research_tools.py
```

### Task 11: SearchBackend Protocol and Data Models

**Files:**
- Create: `src/ave/tools/search.py`
- Test: `tests/test_tools/test_search_protocol.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_tools/test_search_protocol.py
from __future__ import annotations

from ave.tools.search import SearchResult, PageContent, VIDEO_EDITING_SOURCES


class TestSearchDataModels:
    def test_search_result_frozen(self):
        r = SearchResult(
            title="Film Grain in Resolve",
            url="https://forum.blackmagicdesign.com/123",
            snippet="Here's how to add film grain...",
            source="forum.blackmagicdesign.com",
        )
        assert r.title == "Film Grain in Resolve"
        with pytest.raises(AttributeError):
            r.title = "changed"

    def test_page_content(self):
        p = PageContent(
            url="https://example.com",
            text="Full page text here",
            headings=["Introduction", "Method"],
        )
        assert len(p.headings) == 2

    def test_video_editing_sources_exist(self):
        assert "forum.blackmagicdesign.com" in VIDEO_EDITING_SOURCES
        assert len(VIDEO_EDITING_SOURCES) >= 5
```

- [ ] **Step 2: Run to verify failure, Step 3: Implement**

```python
# src/ave/tools/search.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str


@dataclass(frozen=True)
class PageContent:
    url: str
    text: str
    headings: list[str]


VIDEO_EDITING_SOURCES = (
    "forum.blackmagicdesign.com",
    "community.frame.io",
    "liftgammagain.com",
    "reddit.com/r/colorgrading",
    "reddit.com/r/VideoEditing",
    "cinematography.com",
    "docs.arri.com",
    "sony.com/en/articles/technical",
)


class SearchBackend(Protocol):
    async def search(
        self, query: str, max_results: int = 10
    ) -> list[SearchResult]: ...

    async def fetch_page(self, url: str) -> PageContent: ...
```

- [ ] **Step 4: Run test, Step 5: Commit**

```bash
git commit -m "feat(search): add SearchBackend protocol and data models"
```

---

### Task 12: Brave Search Backend

**Files:**
- Create: `src/ave/tools/search_brave.py`
- Test: `tests/test_tools/test_search_brave.py`

- [ ] **Step 1: Write failing test (mocked HTTP)**

```python
# tests/test_tools/test_search_brave.py
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch
from ave.tools.search_brave import BraveSearchBackend


@pytest.mark.asyncio
class TestBraveSearchBackend:
    async def test_search_returns_results(self):
        mock_response = {
            "web": {
                "results": [
                    {
                        "title": "Film Grain Tutorial",
                        "url": "https://example.com/grain",
                        "description": "How to add grain",
                    }
                ]
            }
        }
        backend = BraveSearchBackend(api_key="test-key")
        with patch.object(backend, "_get_json", new_callable=AsyncMock, return_value=mock_response):
            results = await backend.search("film grain davinci resolve")
        assert len(results) == 1
        assert results[0].title == "Film Grain Tutorial"

    async def test_search_with_source_bias(self):
        mock_response = {"web": {"results": []}}
        backend = BraveSearchBackend(api_key="test-key")
        with patch.object(backend, "_get_json", new_callable=AsyncMock, return_value=mock_response) as mock:
            await backend.search(
                "film grain", source_bias=["forum.blackmagicdesign.com"]
            )
            call_args = mock.call_args
            # Verify bias was added to query
            assert "site:forum.blackmagicdesign.com" in str(call_args)

    async def test_missing_api_key_raises(self):
        with pytest.raises(ValueError, match="API key"):
            BraveSearchBackend(api_key="")
```

- [ ] **Step 2-3: Implement BraveSearchBackend**

```python
# src/ave/tools/search_brave.py
from __future__ import annotations

import aiohttp
from ave.tools.search import SearchResult, PageContent


class BraveSearchBackend:
    """Brave Search API backend."""

    _BASE_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("Brave Search API key required")
        self._api_key = api_key

    async def search(
        self,
        query: str,
        max_results: int = 10,
        source_bias: list[str] | None = None,
    ) -> list[SearchResult]:
        if source_bias:
            site_query = " OR ".join(f"site:{s}" for s in source_bias[:3])
            query = f"{query} ({site_query})"

        data = await self._get_json(query, max_results)
        results = []
        for item in data.get("web", {}).get("results", [])[:max_results]:
            url = item.get("url", "")
            results.append(SearchResult(
                title=item.get("title", ""),
                url=url,
                snippet=item.get("description", ""),
                source=url.split("/")[2] if "/" in url else "",
            ))
        return results

    async def fetch_page(self, url: str) -> PageContent:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                text = await resp.text()
        # Basic extraction — strip HTML tags
        import re
        clean = re.sub(r"<[^>]+>", " ", text)
        clean = re.sub(r"\s+", " ", clean).strip()
        headings = re.findall(r"<h[1-6][^>]*>(.*?)</h[1-6]>", text)
        return PageContent(url=url, text=clean[:10000], headings=headings)

    async def _get_json(self, query: str, count: int) -> dict:
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self._api_key,
        }
        params = {"q": query, "count": str(count)}
        async with aiohttp.ClientSession() as session:
            async with session.get(
                self._BASE_URL, headers=headers, params=params
            ) as resp:
                return await resp.json()
```

- [ ] **Step 4: Run test, Step 5: Commit**

```bash
git commit -m "feat(search): add BraveSearchBackend implementation"
```

---

### Task 13: ResearchBrief Data Model and Researcher Logic

**Files:**
- Create: `src/ave/agent/researcher.py`
- Test: `tests/test_agent/test_researcher.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_agent/test_researcher.py
from __future__ import annotations

from ave.agent.researcher import Approach, ResearchBrief, synthesize_research


class TestResearchBrief:
    def test_approach_creation(self):
        a = Approach(
            name="ARRI Official LUT",
            description="Download ARRI's official LogC4 to Rec.709 LUT",
            tool_mapping="Use lut_apply tool with downloaded LUT file",
            source="https://docs.arri.com/luts",
            trade_offs="Accurate but no creative control",
        )
        assert a.name == "ARRI Official LUT"

    def test_research_brief_creation(self):
        brief = ResearchBrief(
            question="How to match ARRI LogC4 to Rec.709?",
            approaches=[
                Approach("LUT", "Use LUT", "lut_apply", "url", "simple"),
            ],
            sources=["https://example.com"],
            confidence=0.8,
        )
        assert len(brief.approaches) == 1
        assert brief.confidence == 0.8

    def test_synthesize_from_search_results(self):
        """synthesize_research extracts approaches from raw search text."""
        from ave.tools.search import SearchResult

        results = [
            SearchResult(
                title="LogC4 to Rec.709 Guide",
                url="https://example.com/1",
                snippet="Use the official ARRI LUT pack for LogC4 conversion",
                source="example.com",
            ),
            SearchResult(
                title="Manual Grade LogC4",
                url="https://example.com/2",
                snippet="Apply a manual S-curve and saturation boost",
                source="example.com",
            ),
        ]
        brief = synthesize_research(
            question="ARRI LogC4 to Rec.709",
            results=results,
            page_contents=[],
        )
        assert brief.question == "ARRI LogC4 to Rec.709"
        assert len(brief.approaches) >= 1
        assert len(brief.sources) == 2
```

- [ ] **Step 2: Run to verify failure, Step 3: Implement**

```python
# src/ave/agent/researcher.py
from __future__ import annotations

from dataclasses import dataclass, field

from ave.tools.search import SearchResult, PageContent


@dataclass(frozen=True)
class Approach:
    name: str
    description: str
    tool_mapping: str
    source: str
    trade_offs: str


@dataclass(frozen=True)
class ResearchBrief:
    question: str
    approaches: list[Approach]
    sources: list[str]
    confidence: float


def synthesize_research(
    question: str,
    results: list[SearchResult],
    page_contents: list[PageContent],
) -> ResearchBrief:
    """Synthesize search results into structured approaches.

    This is the non-LLM path — extracts approaches heuristically.
    The LLM-powered path uses the researcher subagent via Anthropic API.
    """
    approaches: list[Approach] = []
    sources: list[str] = []

    for result in results:
        sources.append(result.url)
        if result.snippet:
            approaches.append(Approach(
                name=result.title[:80],
                description=result.snippet,
                tool_mapping="",  # Filled by LLM in agentic path
                source=result.url,
                trade_offs="",
            ))

    # Deduplicate by name
    seen: set[str] = set()
    unique: list[Approach] = []
    for a in approaches:
        if a.name not in seen:
            seen.add(a.name)
            unique.append(a)

    confidence = min(1.0, len(unique) * 0.3) if unique else 0.0

    return ResearchBrief(
        question=question,
        approaches=unique[:3],
        sources=sources,
        confidence=confidence,
    )
```

- [ ] **Step 4: Run test, Step 5: Commit**

```bash
git commit -m "feat(research): add ResearchBrief model and synthesis logic"
```

---

### Task 14: Researcher Role and Tool Registration

**Files:**
- Modify: `src/ave/agent/roles.py`
- Create: `src/ave/agent/tools/research.py`
- Test: `tests/test_agent/test_research_tools.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_agent/test_research_tools.py
from __future__ import annotations

from ave.agent.registry import ToolRegistry
from ave.agent.tools.research import register_research_tools
from ave.agent.roles import RESEARCHER_ROLE


class TestResearchTools:
    def test_register_research_tools(self):
        reg = ToolRegistry()
        register_research_tools(reg)
        results = reg.search_tools("web search")
        assert any("web_search" in r.name for r in results)

    def test_researcher_role_exists(self):
        assert RESEARCHER_ROLE.name == "Researcher"
        assert "research" in RESEARCHER_ROLE.domains
```

- [ ] **Step 2: Run to verify failure, Step 3: Implement**

Add `RESEARCHER_ROLE` to `roles.py`:
```python
RESEARCHER_ROLE = AgentRole(
    name="Researcher",
    description="Searches the web for video editing techniques, codec information, camera profiles, and forum discussions. Produces structured ResearchBriefs.",
    system_prompt="...",  # Research-focused prompt
    domains=("research",),
)
```

Create `src/ave/agent/tools/research.py`:
```python
def register_research_tools(registry: ToolRegistry) -> None:
    @registry.tool(
        domain="research",
        tags=["web", "search", "forum", "technique", "lookup"],
    )
    def web_search(query: str, sources: list[str] | None = None) -> dict:
        """Search the web for video editing techniques and information."""
        # Returns search results — actual execution handled by session
        return {"query": query, "sources": sources or [], "results": []}

    @registry.tool(
        domain="research",
        tags=["research", "technique", "approach", "forum"],
    )
    def research_technique(question: str) -> dict:
        """Research a video editing technique. Searches web, reads forums,
        synthesizes findings into a ResearchBrief with 1-3 approaches."""
        return {"question": question, "status": "requires_async_execution"}
```

- [ ] **Step 4: Run test, Step 5: Commit**

```bash
git commit -m "feat(research): add Researcher role and research domain tools"
```

---

## Chunk 4: AI Rotoscoping / Keying

### File Structure

```
src/ave/tools/rotoscope.py           # RotoscopeBackend protocol, data models
src/ave/tools/rotoscope_chroma.py    # ChromaKeyBackend
src/ave/tools/rotoscope_sam2.py      # Sam2Backend
src/ave/tools/rotoscope_rvm.py       # RvmBackend
src/ave/tools/mask_eval.py           # MaskEvaluator, MaskQuality
src/ave/agent/tools/vfx.py           # Register VFX domain tools
src/ave/agent/roles.py               # MODIFY: add VFX_ARTIST_ROLE
tests/test_tools/test_rotoscope.py
tests/test_tools/test_chroma_key.py
tests/test_tools/test_mask_eval.py
tests/test_agent/test_vfx_tools.py
```

### Task 15: RotoscopeBackend Protocol and Data Models

**Files:**
- Create: `src/ave/tools/rotoscope.py`
- Test: `tests/test_tools/test_rotoscope.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_tools/test_rotoscope.py
from __future__ import annotations

import numpy as np
import pytest
from ave.tools.rotoscope import (
    SegmentPrompt,
    SegmentationMask,
    MaskCorrection,
)


class TestSegmentPrompt:
    def test_point_prompt(self):
        p = SegmentPrompt(kind="point", value=(100, 200))
        assert p.kind == "point"
        assert p.value == (100, 200)

    def test_box_prompt(self):
        p = SegmentPrompt(kind="box", value=(10, 20, 300, 400))
        assert p.kind == "box"

    def test_text_prompt(self):
        p = SegmentPrompt(kind="text", value="the person in the red shirt")
        assert p.kind == "text"


class TestSegmentationMask:
    def test_create_binary_mask(self):
        mask_data = np.zeros((480, 640), dtype=np.uint8)
        mask_data[100:300, 200:500] = 255
        m = SegmentationMask(
            mask=mask_data, confidence=0.95, frame_index=0, metadata={}
        )
        assert m.confidence == 0.95
        assert m.mask.shape == (480, 640)

    def test_create_alpha_mask(self):
        mask_data = np.zeros((480, 640), dtype=np.float32)
        mask_data[100:300, 200:500] = 0.8
        m = SegmentationMask(
            mask=mask_data, confidence=0.9, frame_index=42, metadata={}
        )
        assert m.mask.dtype == np.float32


class TestMaskCorrection:
    def test_include_point(self):
        c = MaskCorrection(kind="include_point", value=(150, 250))
        assert c.kind == "include_point"

    def test_exclude_region(self):
        c = MaskCorrection(kind="exclude_region", value=(0, 0, 100, 100))
        assert c.kind == "exclude_region"
```

- [ ] **Step 2: Run to verify failure, Step 3: Implement**

```python
# src/ave/tools/rotoscope.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Literal, Protocol

import numpy as np


@dataclass(frozen=True)
class SegmentPrompt:
    kind: Literal["point", "box", "text"]
    value: Any


@dataclass
class SegmentationMask:
    mask: np.ndarray
    confidence: float
    frame_index: int
    metadata: dict


@dataclass(frozen=True)
class MaskCorrection:
    kind: Literal["include_point", "exclude_point", "include_region", "exclude_region"]
    value: Any


class RotoscopeBackend(Protocol):
    def segment_frame(
        self, frame: np.ndarray, prompts: list[SegmentPrompt]
    ) -> SegmentationMask: ...

    def segment_video(
        self, frames: Iterator[np.ndarray], prompts: list[SegmentPrompt],
        keyframes: list[int] | None = None,
    ) -> Iterator[SegmentationMask]: ...

    def refine_mask(
        self, frame: np.ndarray, mask: SegmentationMask,
        corrections: list[MaskCorrection],
    ) -> SegmentationMask: ...
```

- [ ] **Step 4: Run test, Step 5: Commit**

```bash
git commit -m "feat(rotoscope): add RotoscopeBackend protocol and data models"
```

---

### Task 16: MaskEvaluator (Quality Assessment)

**Files:**
- Create: `src/ave/tools/mask_eval.py`
- Test: `tests/test_tools/test_mask_eval.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_tools/test_mask_eval.py
from __future__ import annotations

import numpy as np
import pytest
from ave.tools.mask_eval import MaskQuality, MaskEvaluator
from ave.tools.rotoscope import SegmentationMask


class TestMaskEvaluator:
    def _make_mask(self, frame_idx: int, fill_region: tuple = (100, 200, 300, 500)) -> SegmentationMask:
        data = np.zeros((480, 640), dtype=np.float32)
        y1, x1, y2, x2 = fill_region
        data[y1:y2, x1:x2] = 1.0
        return SegmentationMask(
            mask=data, confidence=0.9, frame_index=frame_idx, metadata={}
        )

    def test_perfect_masks_score_high(self):
        masks = [self._make_mask(i) for i in range(5)]
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
        evaluator = MaskEvaluator()
        quality = evaluator.evaluate(masks, frames)
        assert quality.temporal_stability > 0.9
        assert quality.confidence_mean == pytest.approx(0.9)
        assert len(quality.problem_frames) == 0

    def test_inconsistent_masks_flag_problems(self):
        masks = [self._make_mask(i) for i in range(5)]
        # Make frame 3 wildly different
        masks[3] = SegmentationMask(
            mask=np.ones((480, 640), dtype=np.float32),
            confidence=0.3, frame_index=3, metadata={},
        )
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
        evaluator = MaskEvaluator()
        quality = evaluator.evaluate(masks, frames)
        assert 3 in quality.problem_frames
        assert quality.temporal_stability < 0.9

    def test_empty_masks_list(self):
        evaluator = MaskEvaluator()
        quality = evaluator.evaluate([], [])
        assert quality.confidence_mean == 0.0
```

- [ ] **Step 2: Run to verify failure, Step 3: Implement**

```python
# src/ave/tools/mask_eval.py
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ave.tools.rotoscope import SegmentationMask


@dataclass(frozen=True)
class MaskQuality:
    edge_smoothness: float
    temporal_stability: float
    coverage_ratio: float
    confidence_mean: float
    problem_frames: list[int] = field(default_factory=list)


class MaskEvaluator:
    """Evaluate segmentation mask quality across keyframes."""

    def __init__(self, quality_threshold: float = 0.6) -> None:
        self._threshold = quality_threshold

    def evaluate(
        self, masks: list[SegmentationMask], frames: list[np.ndarray]
    ) -> MaskQuality:
        if not masks:
            return MaskQuality(
                edge_smoothness=0.0, temporal_stability=0.0,
                coverage_ratio=0.0, confidence_mean=0.0,
            )

        confidences = [m.confidence for m in masks]
        confidence_mean = sum(confidences) / len(confidences)

        # Temporal stability: IoU between consecutive masks
        ious: list[float] = []
        for i in range(len(masks) - 1):
            iou = self._mask_iou(masks[i].mask, masks[i + 1].mask)
            ious.append(iou)
        temporal_stability = sum(ious) / len(ious) if ious else 1.0

        # Edge smoothness: gradient magnitude at mask boundary
        smoothness_scores: list[float] = []
        for m in masks:
            smoothness_scores.append(self._edge_smoothness(m.mask))
        edge_smoothness = (
            sum(smoothness_scores) / len(smoothness_scores)
            if smoothness_scores else 0.0
        )

        # Coverage ratio
        coverage_scores: list[float] = []
        for m in masks:
            total = m.mask.size
            fg = np.count_nonzero(m.mask > 0.5)
            coverage_scores.append(fg / total if total > 0 else 0.0)
        coverage_ratio = (
            sum(coverage_scores) / len(coverage_scores)
            if coverage_scores else 0.0
        )

        # Identify problem frames
        problem_frames: list[int] = []
        for i, m in enumerate(masks):
            if m.confidence < self._threshold:
                problem_frames.append(m.frame_index)
        for i, iou in enumerate(ious):
            if iou < self._threshold:
                problem_frames.append(masks[i + 1].frame_index)

        return MaskQuality(
            edge_smoothness=edge_smoothness,
            temporal_stability=temporal_stability,
            coverage_ratio=coverage_ratio,
            confidence_mean=confidence_mean,
            problem_frames=sorted(set(problem_frames)),
        )

    @staticmethod
    def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
        a_bin = a > 0.5
        b_bin = b > 0.5
        intersection = np.logical_and(a_bin, b_bin).sum()
        union = np.logical_or(a_bin, b_bin).sum()
        return float(intersection / union) if union > 0 else 1.0

    @staticmethod
    def _edge_smoothness(mask: np.ndarray) -> float:
        """Higher = smoother edges. Uses gradient magnitude at boundary."""
        binary = (mask > 0.5).astype(np.float32)
        gy = np.diff(binary, axis=0)
        gx = np.diff(binary, axis=1)
        # Count edge pixels vs total boundary length
        edge_pixels = (np.abs(gy).sum() + np.abs(gx).sum())
        if edge_pixels == 0:
            return 1.0
        # Smoothness inversely related to edge complexity
        h, w = mask.shape
        perimeter_estimate = edge_pixels
        # Compare to ideal rectangle perimeter
        ideal = 2 * (h + w)
        return float(min(1.0, ideal / (perimeter_estimate + 1)))
```

- [ ] **Step 4: Run test, Step 5: Commit**

```bash
git commit -m "feat(rotoscope): add MaskEvaluator with IoU and edge analysis"
```

---

### Task 17: ChromaKeyBackend

**Files:**
- Create: `src/ave/tools/rotoscope_chroma.py`
- Test: `tests/test_tools/test_chroma_key.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_tools/test_chroma_key.py
from __future__ import annotations

import numpy as np
import pytest
from ave.tools.rotoscope_chroma import ChromaKeyBackend
from ave.tools.rotoscope import SegmentPrompt


class TestChromaKeyBackend:
    def test_green_screen_keying(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Green background
        frame[:, :] = [0, 255, 0]
        # White foreground in center
        frame[30:70, 30:70] = [255, 255, 255]

        backend = ChromaKeyBackend()
        mask = backend.segment_frame(
            frame, [SegmentPrompt(kind="text", value="green")]
        )
        # Center should be foreground (1), edges should be background (0)
        assert mask.mask[50, 50] > 0.5  # foreground
        assert mask.mask[5, 5] < 0.5    # green = keyed out

    def test_blue_screen_keying(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :] = [255, 0, 0]  # Blue (BGR)
        frame[30:70, 30:70] = [255, 255, 255]

        backend = ChromaKeyBackend()
        mask = backend.segment_frame(
            frame, [SegmentPrompt(kind="text", value="blue")]
        )
        assert mask.mask[50, 50] > 0.5
        assert mask.mask[5, 5] < 0.5

    def test_adjustable_tolerance(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :] = [0, 200, 0]  # Slightly dark green

        backend = ChromaKeyBackend(tolerance=0.1)
        mask_tight = backend.segment_frame(
            frame, [SegmentPrompt(kind="text", value="green")]
        )

        backend_loose = ChromaKeyBackend(tolerance=0.5)
        mask_loose = backend_loose.segment_frame(
            frame, [SegmentPrompt(kind="text", value="green")]
        )
        # Loose tolerance should key out more
        assert np.sum(mask_loose.mask < 0.5) >= np.sum(mask_tight.mask < 0.5)
```

- [ ] **Step 2: Run to verify failure, Step 3: Implement**

```python
# src/ave/tools/rotoscope_chroma.py
from __future__ import annotations

from typing import Iterator

import numpy as np

from ave.tools.rotoscope import (
    MaskCorrection,
    RotoscopeBackend,
    SegmentationMask,
    SegmentPrompt,
)

_KEY_COLORS = {
    "green": np.array([0, 255, 0], dtype=np.float32),
    "blue": np.array([255, 0, 0], dtype=np.float32),  # BGR
}


class ChromaKeyBackend:
    """Deterministic chroma keying — not ML-based."""

    def __init__(
        self,
        tolerance: float = 0.3,
        spill_suppression: float = 0.5,
    ) -> None:
        self._tolerance = tolerance
        self._spill = spill_suppression

    def segment_frame(
        self, frame: np.ndarray, prompts: list[SegmentPrompt]
    ) -> SegmentationMask:
        key_color_name = "green"
        for p in prompts:
            if p.kind == "text" and isinstance(p.value, str):
                val = p.value.lower()
                if "blue" in val:
                    key_color_name = "blue"
                elif "green" in val:
                    key_color_name = "green"

        key_color = _KEY_COLORS[key_color_name]
        frame_f = frame.astype(np.float32)
        diff = np.linalg.norm(frame_f - key_color, axis=2) / 441.67  # max dist
        # Pixels close to key color → background (0), far → foreground (1)
        threshold = self._tolerance
        mask = np.where(diff > threshold, 1.0, 0.0).astype(np.float32)

        return SegmentationMask(
            mask=mask,
            confidence=0.95,
            frame_index=0,
            metadata={"key_color": key_color_name, "tolerance": self._tolerance},
        )

    def segment_video(
        self, frames: Iterator[np.ndarray], prompts: list[SegmentPrompt],
        keyframes: list[int] | None = None,
    ) -> Iterator[SegmentationMask]:
        for i, frame in enumerate(frames):
            mask = self.segment_frame(frame, prompts)
            mask.frame_index = i
            yield mask

    def refine_mask(
        self, frame: np.ndarray, mask: SegmentationMask,
        corrections: list[MaskCorrection],
    ) -> SegmentationMask:
        refined = mask.mask.copy()
        for c in corrections:
            if c.kind == "include_point" and isinstance(c.value, tuple):
                y, x = c.value
                refined[max(0, y-5):y+5, max(0, x-5):x+5] = 1.0
            elif c.kind == "exclude_point" and isinstance(c.value, tuple):
                y, x = c.value
                refined[max(0, y-5):y+5, max(0, x-5):x+5] = 0.0
        return SegmentationMask(
            mask=refined, confidence=mask.confidence,
            frame_index=mask.frame_index, metadata=mask.metadata,
        )
```

- [ ] **Step 4: Run test, Step 5: Commit**

```bash
git commit -m "feat(rotoscope): add ChromaKeyBackend with green/blue screen support"
```

---

### Task 18: VFX Tools Registration and Role

**Files:**
- Create: `src/ave/agent/tools/vfx.py`
- Modify: `src/ave/agent/roles.py`
- Test: `tests/test_agent/test_vfx_tools.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_agent/test_vfx_tools.py
from __future__ import annotations

from ave.agent.registry import ToolRegistry
from ave.agent.tools.vfx import register_vfx_tools
from ave.agent.roles import VFX_ARTIST_ROLE


class TestVfxTools:
    def test_register_vfx_tools(self):
        reg = ToolRegistry()
        register_vfx_tools(reg)
        results = reg.search_tools("segment")
        assert any("segment_video" in r.name for r in results)

    def test_four_vfx_tools_registered(self):
        reg = ToolRegistry()
        register_vfx_tools(reg)
        vfx_tools = reg.search_tools(domain="vfx")
        names = {r.name for r in vfx_tools}
        assert "segment_video" in str(names)
        assert "refine_mask" in str(names)
        assert "evaluate_mask" in str(names)
        assert "apply_mask" in str(names)

    def test_only_apply_mask_modifies_timeline(self):
        reg = ToolRegistry()
        register_vfx_tools(reg)
        # apply_mask should modify timeline
        assert reg.tool_modifies_timeline("apply_mask")
        # Others should not
        assert not reg.tool_modifies_timeline("segment_video")
        assert not reg.tool_modifies_timeline("evaluate_mask")

    def test_vfx_artist_role_exists(self):
        assert VFX_ARTIST_ROLE.name == "VFX Artist"
        assert "vfx" in VFX_ARTIST_ROLE.domains
```

- [ ] **Step 2: Run to verify failure, Step 3: Implement**

Add to `roles.py`:
```python
VFX_ARTIST_ROLE = AgentRole(
    name="VFX Artist",
    description="Handles rotoscoping, keying, segmentation, and visual effects compositing. Uses the keyframe feedback loop to iteratively refine masks.",
    system_prompt="...",  # VFX-focused prompt
    domains=("vfx",),
)
```

Create `src/ave/agent/tools/vfx.py`:
```python
def register_vfx_tools(registry: ToolRegistry) -> None:
    @registry.tool(
        domain="vfx",
        tags=["segment", "rotoscope", "mask", "matte", "key"],
    )
    def segment_video(
        asset_path: str, prompts: list[dict],
        backend: str = "auto",
    ) -> dict:
        """Segment objects in video. Returns mask asset path."""
        return {"asset_path": asset_path, "prompts": prompts, "backend": backend}

    @registry.tool(
        domain="vfx",
        tags=["refine", "mask", "correction", "fix"],
    )
    def refine_mask(
        mask_path: str, corrections: list[dict],
        frames: list[int] | None = None,
    ) -> dict:
        """Refine segmentation mask at specific frames."""
        return {"mask_path": mask_path, "corrections": corrections}

    @registry.tool(
        domain="vfx",
        tags=["evaluate", "quality", "mask", "check"],
    )
    def evaluate_mask(mask_path: str, asset_path: str) -> dict:
        """Assess mask quality. Returns MaskQuality metrics."""
        return {"mask_path": mask_path, "asset_path": asset_path}

    @registry.tool(
        domain="vfx",
        modifies_timeline=True,
        tags=["apply", "mask", "composite", "remove background", "key"],
    )
    def apply_mask(
        clip_id: str, mask_path: str,
        operation: str = "remove_background",
    ) -> dict:
        """Apply mask to timeline clip. Operations: remove_background, composite, replace."""
        return {"clip_id": clip_id, "mask_path": mask_path, "operation": operation}
```

- [ ] **Step 4: Run test, Step 5: Commit**

```bash
git commit -m "feat(vfx): add VFX Artist role and rotoscope/keying tools"
```

---

### Task 19: SAM 2 and RVM Backend Stubs

**Files:**
- Create: `src/ave/tools/rotoscope_sam2.py`, `src/ave/tools/rotoscope_rvm.py`
- Test: `tests/test_tools/test_rotoscope_backends.py`

These are stub implementations that verify the protocol contract. Real ML inference requires GPU and large model downloads, so the stubs return synthetic masks for testing. The real implementations will be filled in when running inside the Docker container with CUDA.

- [ ] **Step 1: Write failing test**

```python
# tests/test_tools/test_rotoscope_backends.py
from __future__ import annotations

import numpy as np
import pytest
from ave.tools.rotoscope import SegmentPrompt, RotoscopeBackend


class TestSam2BackendProtocol:
    def test_implements_protocol(self):
        from ave.tools.rotoscope_sam2 import Sam2Backend
        # Verify it satisfies the Protocol (structural typing)
        backend: RotoscopeBackend = Sam2Backend(model_size="tiny")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mask = backend.segment_frame(
            frame, [SegmentPrompt(kind="point", value=(320, 240))]
        )
        assert mask.mask.shape == (480, 640)
        assert 0.0 <= mask.confidence <= 1.0


class TestRvmBackendProtocol:
    def test_implements_protocol(self):
        from ave.tools.rotoscope_rvm import RvmBackend
        backend: RotoscopeBackend = RvmBackend()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mask = backend.segment_frame(frame, [])
        assert mask.mask.shape == (480, 640)
        assert mask.mask.dtype == np.float32  # Soft alpha matte
```

- [ ] **Step 2: Run to verify failure, Step 3: Implement stubs**

```python
# src/ave/tools/rotoscope_sam2.py
from __future__ import annotations

from typing import Iterator

import numpy as np

from ave.tools.rotoscope import (
    MaskCorrection, SegmentationMask, SegmentPrompt,
)


class Sam2Backend:
    """SAM 2 video segmentation backend.

    Requires: torch, segment-anything-2
    Real implementation loads model on first use.
    This stub returns synthetic masks for testing.
    """

    def __init__(self, model_size: str = "large") -> None:
        self._model_size = model_size
        self._model = None  # Lazy loaded

    def segment_frame(
        self, frame: np.ndarray, prompts: list[SegmentPrompt]
    ) -> SegmentationMask:
        h, w = frame.shape[:2]
        # Stub: return center rectangle mask
        mask = np.zeros((h, w), dtype=np.float32)
        mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 1.0
        return SegmentationMask(
            mask=mask, confidence=0.85, frame_index=0,
            metadata={"backend": "sam2", "model": self._model_size},
        )

    def segment_video(
        self, frames: Iterator[np.ndarray], prompts: list[SegmentPrompt],
        keyframes: list[int] | None = None,
    ) -> Iterator[SegmentationMask]:
        for i, frame in enumerate(frames):
            mask = self.segment_frame(frame, prompts)
            mask.frame_index = i
            yield mask

    def refine_mask(
        self, frame: np.ndarray, mask: SegmentationMask,
        corrections: list[MaskCorrection],
    ) -> SegmentationMask:
        refined = mask.mask.copy()
        return SegmentationMask(
            mask=refined, confidence=min(1.0, mask.confidence + 0.05),
            frame_index=mask.frame_index, metadata=mask.metadata,
        )
```

```python
# src/ave/tools/rotoscope_rvm.py
# Same pattern but returns soft alpha mattes (float32)
# and doesn't require prompts (assumes human subject)
```

- [ ] **Step 4: Run test, Step 5: Commit**

```bash
git commit -m "feat(rotoscope): add SAM 2 and RVM backend stubs"
```

---

### Task 20: Rotoscope Plugin Packaging

**Files:**
- Create: `src/ave/plugins/vfx-rotoscope/plugin.yaml`
- Create: `src/ave/plugins/vfx-rotoscope/__init__.py`
- Create: `src/ave/plugins/vfx-rotoscope/skills/auto-rotoscope.md`
- Create: `src/ave/plugins/vfx-rotoscope/skills/guided-rotoscope.md`
- Test: `tests/test_plugins/test_rotoscope_plugin.py`

- [ ] **Step 1: Write failing test for plugin loading**

```python
# tests/test_plugins/test_rotoscope_plugin.py
from __future__ import annotations

from pathlib import Path
from ave.plugins.discovery import parse_manifest


class TestRotoscopePlugin:
    def test_manifest_parses(self):
        manifest_path = (
            Path(__file__).resolve().parents[2]
            / "src" / "ave" / "plugins" / "vfx-rotoscope" / "plugin.yaml"
        )
        manifest = parse_manifest(manifest_path)
        assert manifest.name == "vfx-rotoscope"
        assert manifest.domain == "vfx"
        assert len(manifest.tools) == 4

    def test_plugin_has_register_function(self):
        from ave.plugins import vfx_rotoscope  # noqa — will be renamed on import
        # Actually import directly
        import importlib.util
        init_path = (
            Path(__file__).resolve().parents[2]
            / "src" / "ave" / "plugins" / "vfx-rotoscope" / "__init__.py"
        )
        spec = importlib.util.spec_from_file_location("vfx_roto", init_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "register")
```

- [ ] **Step 2: Run to verify failure, Step 3: Create plugin structure**

```yaml
# src/ave/plugins/vfx-rotoscope/plugin.yaml
name: vfx-rotoscope
description: AI rotoscoping and keying with SAM 2, RVM, and chroma key backends
version: 1.0.0
domain: vfx
tools:
  - name: segment_video
    summary: Segment objects in video using AI or chroma key
  - name: refine_mask
    summary: Refine segmentation mask at specific frames
  - name: evaluate_mask
    summary: Assess mask quality metrics
  - name: apply_mask
    summary: Apply mask to timeline clip
requires:
  python: [numpy]
  system: []
```

```python
# src/ave/plugins/vfx-rotoscope/__init__.py
from __future__ import annotations

def register(registry, namespace="ave"):
    from ave.agent.tools.vfx import register_vfx_tools
    register_vfx_tools(registry)
```

```markdown
# src/ave/plugins/vfx-rotoscope/skills/auto-rotoscope.md
---
name: auto-rotoscope
description: Automatically segment and extract subject from video
domain: vfx
triggers:
  - rotoscope
  - remove background
  - extract subject
  - isolate person
  - cut out
---

## Auto-Rotoscope Workflow

1. Analyze the clip to identify keyframes (scene cuts, motion peaks)
2. Run segmentation on keyframes using the best available backend
3. Evaluate mask quality (edge smoothness, temporal stability)
4. If quality < threshold, refine problem frames and re-evaluate
5. Once keyframes pass quality check, propagate to full clip
6. Run final quality check on sampled frames
7. Apply mask to timeline

## Backend Selection
- If clip has green/blue screen → use chroma key (fastest, most reliable)
- If subject is human and scene is simple → use RVM (fast, good edges)
- Otherwise → use SAM 2 (highest quality, slowest)
```

```markdown
# src/ave/plugins/vfx-rotoscope/skills/guided-rotoscope.md
---
name: guided-rotoscope
description: Human-guided rotoscoping with approval checkpoints
domain: vfx
triggers:
  - guided rotoscope
  - manual rotoscope
  - review mask
  - check rotoscope
---

## Guided Rotoscope Workflow

1. Analyze the clip to identify keyframes
2. Run initial segmentation on keyframes
3. **CHECKPOINT**: Present keyframe masks to user for review
4. User approves or flags problem areas
5. If flagged: refine specific frames based on user feedback
6. **CHECKPOINT**: Present refined masks
7. Once approved: propagate to full clip
8. **CHECKPOINT**: Present sampled frames from full result
9. Once approved: apply mask to timeline
```

- [ ] **Step 4: Run test, Step 5: Commit**

```bash
git commit -m "feat(rotoscope): package as built-in plugin with auto/guided skills"
```

---

## Chunk 5: Integration and E2E Tests

### Task 21: Wire Plugins and Skills into Session Startup

**Files:**
- Modify: `src/ave/agent/session.py`
- Test: `tests/test_agent/test_session.py`

- [ ] **Step 1: Add default plugin/skill directory resolution**

```python
# In EditingSession.__init__ or a class method:
@classmethod
def default_plugin_dirs(cls) -> list[Path]:
    """Plugin search directories in priority order."""
    dirs = []
    # Project-specific
    cwd = Path.cwd()
    if (cwd / ".ave" / "plugins").is_dir():
        dirs.append(cwd / ".ave" / "plugins")
    # User
    home = Path.home() / ".ave" / "plugins"
    if home.is_dir():
        dirs.append(home)
    # Built-in
    builtin = Path(__file__).parent.parent / "plugins"
    if builtin.is_dir():
        dirs.append(builtin)
    return dirs

@classmethod
def default_skill_dirs(cls) -> list[Path]:
    """Skill search directories in priority order."""
    dirs = []
    cwd = Path.cwd()
    if (cwd / ".ave" / "skills").is_dir():
        dirs.append(cwd / ".ave" / "skills")
    home = Path.home() / ".ave" / "skills"
    if home.is_dir():
        dirs.append(home)
    builtin = Path(__file__).parent.parent / "skills"
    if builtin.is_dir():
        dirs.append(builtin)
    return dirs
```

- [ ] **Step 2: Write test verifying built-in plugin discovery**

```python
def test_session_discovers_builtin_rotoscope_plugin():
    session = EditingSession()
    results = session.search_tools("segment")
    # Should find vfx-rotoscope plugin tools
    assert any("segment" in r.name for r in results)
```

- [ ] **Step 3: Run test, fix, commit**

```bash
git commit -m "feat(session): wire default plugin/skill directory resolution"
```

---

### Task 22: MCP Server E2E Test

**Files:**
- Create: `tests/test_mcp/test_e2e.py`

- [ ] **Step 1: Write E2E test (non-LLM)**

```python
# tests/test_mcp/test_e2e.py
from __future__ import annotations

import pytest
from ave.mcp.server import create_mcp_server


class TestMcpE2E:
    def test_search_then_call_workflow(self):
        """Simulate a consuming agent's workflow: search → schema → call."""
        server = create_mcp_server()

        # 1. Search for trim tools
        results = server.call_tool("search_tools", {"query": "trim"})
        assert len(results) >= 1
        tool_name = results[0]["name"]

        # 2. Get project state
        state = server.call_tool("get_project_state", {})
        assert "clip_count" in state

    def test_all_six_tools_callable(self):
        """Verify all 6 MCP tools respond without error."""
        server = create_mcp_server()
        server.call_tool("get_project_state", {})
        server.call_tool("search_tools", {"query": "trim"})
        server.call_tool("render_preview", {})
        server.call_tool("ingest_asset", {"path": "/tmp/test.mov"})
        server.call_tool("edit_video", {"instruction": "test"})
        server.call_tool("call_tool", {"name": "trim", "params": {}})
```

- [ ] **Step 2: Write LLM-powered E2E test**

```python
@pytest.mark.llm
class TestMcpLlmE2E:
    def test_llm_can_use_mcp_tools(self):
        """Real LLM call: give Claude the 6 MCP tool schemas, ask it to
        search for a video editing tool, verify it calls search_tools."""
        # Uses ANTHROPIC_API_KEY
        import anthropic
        client = anthropic.Anthropic()

        server = create_mcp_server()
        tools = [
            {"name": t.name, "description": t.description,
             "input_schema": t.input_schema}
            for t in server.list_tools()
        ]

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            tools=tools,
            messages=[{
                "role": "user",
                "content": "Find me tools related to trimming video clips",
            }],
        )
        # Claude should call search_tools
        tool_calls = [b for b in response.content if b.type == "tool_use"]
        assert len(tool_calls) >= 1
        assert tool_calls[0].name == "search_tools"
```

- [ ] **Step 3: Run tests, commit**

```bash
git commit -m "test(mcp): add E2E tests including LLM-powered tool discovery"
```

---

### Task 23: Research Agent E2E Test

**Files:**
- Create: `tests/test_agent/test_research_e2e.py`

- [ ] **Step 1: Write mocked E2E test**

```python
# tests/test_agent/test_research_e2e.py
from __future__ import annotations

from unittest.mock import AsyncMock, patch
import pytest
from ave.agent.researcher import synthesize_research
from ave.tools.search import SearchResult


class TestResearchE2E:
    def test_synthesize_from_multiple_sources(self):
        results = [
            SearchResult("LUT Method", "https://a.com", "Use official LUT pack", "a.com"),
            SearchResult("Manual Grade", "https://b.com", "S-curve with lift", "b.com"),
            SearchResult("ACES Pipeline", "https://c.com", "Use ACES IDT transform", "c.com"),
        ]
        brief = synthesize_research("LogC4 to Rec.709", results, [])
        assert len(brief.approaches) == 3
        assert brief.confidence >= 0.8
        assert len(brief.sources) == 3
```

- [ ] **Step 2: Run test, commit**

```bash
git commit -m "test(research): add E2E research synthesis test"
```

---

### Task 24: Rotoscope Feedback Loop E2E Test

**Files:**
- Create: `tests/test_tools/test_rotoscope_e2e.py`

- [ ] **Step 1: Write E2E feedback loop test**

```python
# tests/test_tools/test_rotoscope_e2e.py
from __future__ import annotations

import numpy as np
from ave.tools.rotoscope import SegmentPrompt, MaskCorrection
from ave.tools.rotoscope_sam2 import Sam2Backend
from ave.tools.rotoscope_chroma import ChromaKeyBackend
from ave.tools.mask_eval import MaskEvaluator


class TestFeedbackLoop:
    def test_segment_evaluate_refine_cycle(self):
        """Simulate the agent's feedback loop: segment → evaluate → refine."""
        backend = Sam2Backend(model_size="tiny")
        evaluator = MaskEvaluator(quality_threshold=0.6)

        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
        prompts = [SegmentPrompt(kind="point", value=(320, 240))]

        # Step 1: Segment keyframes
        masks = [backend.segment_frame(f, prompts) for i, f in enumerate(frames)]
        for i, m in enumerate(masks):
            m.frame_index = i

        # Step 2: Evaluate
        quality = evaluator.evaluate(masks, frames)
        assert quality.confidence_mean > 0

        # Step 3: If problems, refine
        if quality.problem_frames:
            for frame_idx in quality.problem_frames:
                masks[frame_idx] = backend.refine_mask(
                    frames[frame_idx], masks[frame_idx],
                    [MaskCorrection(kind="include_point", value=(200, 300))],
                )

        # Step 4: Re-evaluate
        quality_after = evaluator.evaluate(masks, frames)
        assert quality_after.confidence_mean >= quality.confidence_mean

    def test_chroma_key_full_pipeline(self):
        """Green screen clip through full pipeline."""
        backend = ChromaKeyBackend(tolerance=0.3)
        evaluator = MaskEvaluator()

        # Create green screen frames with white subject
        frames = []
        for _ in range(3):
            f = np.zeros((480, 640, 3), dtype=np.uint8)
            f[:, :] = [0, 255, 0]  # Green
            f[100:380, 200:440] = [200, 200, 200]  # Subject
            frames.append(f)

        prompts = [SegmentPrompt(kind="text", value="green")]
        masks = [backend.segment_frame(f, prompts) for f in frames]
        for i, m in enumerate(masks):
            m.frame_index = i

        quality = evaluator.evaluate(masks, frames)
        assert quality.temporal_stability > 0.9  # Consistent frames
        assert quality.confidence_mean > 0.9
```

- [ ] **Step 2: Run test, commit**

```bash
git commit -m "test(rotoscope): add feedback loop E2E tests"
```

---

### Task 25: Final Integration — Update ALL_ROLES and Session

**Files:**
- Modify: `src/ave/agent/roles.py`
- Modify: `src/ave/agent/session.py`
- Modify: `src/ave/agent/multi_agent.py`

- [ ] **Step 1: Add new roles to ALL_ROLES**

```python
# In roles.py:
ALL_ROLES: tuple[AgentRole, ...] = (
    EDITOR_ROLE,
    COLORIST_ROLE,
    SOUND_DESIGNER_ROLE,
    TRANSCRIPTIONIST_ROLE,
    RESEARCHER_ROLE,
    VFX_ARTIST_ROLE,
)
```

- [ ] **Step 2: Register research and vfx tools in session's _load_all_tools**

```python
# In session.py _load_all_tools():
from ave.agent.tools.research import register_research_tools
from ave.agent.tools.vfx import register_vfx_tools
register_research_tools(self._registry)
register_vfx_tools(self._registry)
```

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -x -v --ignore=tests/test_web -m "not gpu and not slow and not llm"`
Expected: All existing + new tests pass

- [ ] **Step 4: Commit**

```bash
git commit -m "feat: integrate research and VFX roles into session and multi-agent"
```

---

### Task 26: Run Full Test Suite and Fix

- [ ] **Step 1: Run everything**

```bash
python -m pytest tests/ -v --tb=short -m "not gpu and not slow and not llm and not requires_ges"
```

- [ ] **Step 2: Fix any failures**

- [ ] **Step 3: Run LLM tests separately**

```bash
python -m pytest tests/ -v -m "llm" --tb=short
```

- [ ] **Step 4: Final commit**

```bash
git commit -m "fix: resolve integration test failures for Phase 9"
```
