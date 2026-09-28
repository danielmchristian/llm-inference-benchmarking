"""Smoke tests: the package installs, imports, and is packaged correctly."""

from importlib import metadata, resources

import llmbench


def test_version_matches_installed_metadata() -> None:
    """__version__ is a static string; it must not drift from pyproject.toml."""
    assert llmbench.__version__ == metadata.version("llm-bench")


def test_py_typed_marker_is_packaged() -> None:
    """Without the PEP 561 marker, downstream mypy ignores our annotations."""
    assert resources.files("llmbench").joinpath("py.typed").is_file()
