"""Tests for ave.tools.aces_config — ACES 2.0 configuration support."""

from __future__ import annotations

import pytest

from ave.tools.aces_config import (
    DEFAULT_WORKING_SPACE,
    DEFAULT_OCIO_CONFIG,
    DEFAULT_ODT,
    AcesConfigError,
    get_builtin_aces_config,
)


# ---------------------------------------------------------------------------
# Pure-logic tests (no PyOpenColorIO required)
# ---------------------------------------------------------------------------


class TestConstants:
    def test_default_working_space(self) -> None:
        assert DEFAULT_WORKING_SPACE == "ACES - ACEScct"

    def test_default_ocio_config_is_nonempty_string(self) -> None:
        assert isinstance(DEFAULT_OCIO_CONFIG, str)
        assert len(DEFAULT_OCIO_CONFIG) > 0

    def test_default_ocio_config_starts_with_ocio_uri(self) -> None:
        assert DEFAULT_OCIO_CONFIG.startswith("ocio://")

    def test_default_odt(self) -> None:
        assert DEFAULT_ODT == "ACES 1.0 - SDR Video - Rec.709"


class TestGetBuiltinAcesConfig:
    def test_returns_string(self) -> None:
        result = get_builtin_aces_config()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_ocio_uri_or_constant(self) -> None:
        result = get_builtin_aces_config()
        # Should always return DEFAULT_OCIO_CONFIG (the URI string)
        assert result == DEFAULT_OCIO_CONFIG


class TestAcesConfigError:
    def test_is_exception(self) -> None:
        assert issubclass(AcesConfigError, Exception)

    def test_can_be_raised(self) -> None:
        with pytest.raises(AcesConfigError, match="test error"):
            raise AcesConfigError("test error")


# ---------------------------------------------------------------------------
# OCIO-dependent tests (skipped if PyOpenColorIO not installed)
# ---------------------------------------------------------------------------


class TestListColorspaces:
    @pytest.fixture(autouse=True)
    def _require_ocio(self) -> None:
        pytest.importorskip("PyOpenColorIO")

    def test_returns_list(self) -> None:
        from ave.tools.aces_config import list_colorspaces

        result = list_colorspaces(None)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_entries_are_strings(self) -> None:
        from ave.tools.aces_config import list_colorspaces

        result = list_colorspaces(None)
        assert all(isinstance(cs, str) for cs in result)

    def test_invalid_config_path_raises(self) -> None:
        from ave.tools.aces_config import list_colorspaces

        with pytest.raises(AcesConfigError):
            list_colorspaces("/no/such/config.ocio")


class TestValidateColorspace:
    @pytest.fixture(autouse=True)
    def _require_ocio(self) -> None:
        pytest.importorskip("PyOpenColorIO")

    def test_valid_colorspace_returns_true(self) -> None:
        from ave.tools.aces_config import validate_colorspace

        assert validate_colorspace("ACES - ACEScct", None) is True

    def test_invalid_colorspace_returns_false(self) -> None:
        from ave.tools.aces_config import validate_colorspace

        assert validate_colorspace("NotARealSpace", None) is False
