"""Tests that the harness module imports cleanly and exposes its version."""


def test_harness_module_imports():
    import ave.harness  # noqa: F401


def test_harness_exports_version_string():
    import ave.harness

    assert isinstance(ave.harness.__version__, str)
    assert len(ave.harness.__version__) > 0
