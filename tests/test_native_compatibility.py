# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Load real shared libraries to verify the native safety contract."""

import shutil
import subprocess
from pathlib import Path

import pytest

from torch_npu_memory_saver import saver


@pytest.fixture(scope="module")
def native_compiler():
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("native compatibility tests require a C++17 compiler")
    return compiler


@pytest.fixture(scope="module")
def current_native_library(tmp_path_factory, native_compiler):
    build = tmp_path_factory.mktemp("current-native")
    library = build / "libascend_memory_saver.so"
    native = Path(__file__).resolve().parents[1] / "native"
    subprocess.run(
        [
            native_compiler,
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-pthread",
            "-DSAVER_FAKE_ACL=1",
            f"-I{native / 'include'}",
            f"-I{native / 'tests'}",
            str(native / "src/ascend_saver.cpp"),
            str(native / "tests/fake_acl.cpp"),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return library


def test_current_native_library_is_accepted(current_native_library):
    backend = saver.NativeBackend(str(current_native_library))
    backend.check_health()
    assert backend.stats()["abi_version"] == 4
    assert backend.stats()["segments"] == 0


@pytest.mark.parametrize("version", [2, 3, 5, None])
def test_override_rejects_incompatible_native_library(
    monkeypatch, tmp_path, native_compiler, version
):
    # Keep the production filename: a path/name check cannot detect a stale build.
    library = tmp_path / "libascend_memory_saver.so"
    source = tmp_path / "legacy.cpp"
    definition = (
        f'extern "C" int saver_abi_version() {{ return {version}; }}'
        if version is not None
        else 'extern "C" int legacy_library() { return 0; }'
    )
    source.write_text(definition)
    subprocess.run(
        [native_compiler, "-shared", "-fPIC", str(source), "-o", str(library)],
        check=True,
        capture_output=True,
        text=True,
    )
    monkeypatch.setenv("TORCH_NPU_MEMORY_SAVER_LIBRARY", str(library))
    with pytest.raises(RuntimeError, match="requires native ABI 4"):
        saver.NativeBackend(saver._library_path())
