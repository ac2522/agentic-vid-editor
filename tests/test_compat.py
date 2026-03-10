"""Tests for ave._compat optional dependency helper."""

import pytest

from ave._compat import import_optional


class TestImportOptional:
    def test_import_stdlib_module(self):
        mod = import_optional("json")
        assert hasattr(mod, "dumps")

    def test_import_missing_module_raises(self):
        with pytest.raises(ImportError, match="Missing optional dependency 'nonexistent_pkg_xyz'"):
            import_optional("nonexistent_pkg_xyz")

    def test_error_message_includes_extra(self):
        with pytest.raises(ImportError, match="pip install ave\\[vision\\]"):
            import_optional("nonexistent_pkg_xyz", extra="vision")

    def test_error_message_uses_mapping(self):
        """Known packages get their mapped extra in the error message."""
        with pytest.raises(ImportError, match="pip install ave\\[scene\\]"):
            import_optional("scenedetect")

    def test_error_message_pip_install_passthrough(self):
        """Extras starting with 'pip install' are used verbatim."""
        with pytest.raises(ImportError, match="pip install pywhispercpp"):
            import_optional("pywhispercpp")

    def test_returns_module(self):
        mod = import_optional("os")
        import os

        assert mod is os
