"""Tests for model download manager."""

from __future__ import annotations

from pathlib import Path

import pytest

from ave.tools.model_manager import ModelInfo, ModelManager, KNOWN_MODELS


class TestModelInfo:
    def test_size_human_gb(self):
        m = ModelInfo(name="big", description="", size_bytes=2_400_000_000, url="")
        assert "2.4 GB" in m.size_human

    def test_size_human_mb(self):
        m = ModelInfo(name="med", description="", size_bytes=800_000_000, url="")
        assert "800 MB" in m.size_human

    def test_known_models_include_sam3(self):
        assert "sam3-large" in KNOWN_MODELS
        assert "sam3-base" in KNOWN_MODELS

    def test_known_models_include_matanyone2(self):
        assert "matanyone2" in KNOWN_MODELS


class TestModelManager:
    def test_model_not_available_initially(self, tmp_path):
        mgr = ModelManager(cache_dir=tmp_path)
        assert not mgr.is_available("sam3-large")

    def test_ensure_model_downloads(self, tmp_path):
        mgr = ModelManager(cache_dir=tmp_path)
        path = mgr.ensure_model("sam3-large")
        assert path.exists()
        assert mgr.is_available("sam3-large")

    def test_ensure_model_uses_cache(self, tmp_path):
        mgr = ModelManager(cache_dir=tmp_path)
        path1 = mgr.ensure_model("sam3-base")
        path2 = mgr.ensure_model("sam3-base")
        assert path1 == path2

    def test_consent_denied_raises(self, tmp_path):
        mgr = ModelManager(
            cache_dir=tmp_path,
            consent_callback=lambda _: False,
        )
        with pytest.raises(RuntimeError, match="consent denied"):
            mgr.ensure_model("sam3-large")

    def test_unknown_model_raises(self, tmp_path):
        mgr = ModelManager(cache_dir=tmp_path)
        with pytest.raises(ValueError, match="Unknown model"):
            mgr.ensure_model("nonexistent-model")
