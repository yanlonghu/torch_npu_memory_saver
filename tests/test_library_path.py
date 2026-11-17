"""Resolve installed libraries and reject unusable explicit overrides."""

import pytest

from torch_npu_memory_saver import saver


def test_installed_library_is_found_without_environment_override(monkeypatch, tmp_path):
    package = tmp_path / "torch_npu_memory_saver"
    library = package / "lib/libascend_memory_saver.so"
    library.parent.mkdir(parents=True)
    library.touch()
    monkeypatch.setattr(saver, "__file__", str(package / "saver.py"))
    monkeypatch.delenv("TORCH_NPU_MEMORY_SAVER_LIBRARY", raising=False)
    assert saver._library_path() == str(library)


def test_explicit_library_overrides_package_location(monkeypatch, tmp_path):
    library = tmp_path / "external.so"
    library.touch()
    monkeypatch.setenv("TORCH_NPU_MEMORY_SAVER_LIBRARY", str(library))
    assert saver._library_path() == str(library)


@pytest.mark.parametrize("kind", ["relative", "missing", "directory"])
def test_bad_override_does_not_silently_load_another_library(
    monkeypatch, tmp_path, kind
):
    path = {
        "relative": "relative.so",
        "missing": str(tmp_path / "missing.so"),
        "directory": str(tmp_path),
    }[kind]
    monkeypatch.setenv("TORCH_NPU_MEMORY_SAVER_LIBRARY", path)
    with pytest.raises(RuntimeError, match="existing absolute saver library path"):
        saver._library_path()
