"""Build against the deployment's installed CANN toolkit."""

from pathlib import Path
import subprocess
from setuptools import Distribution, setup
from setuptools.command.build_py import build_py


class BuildPy(build_py):
    def run(self):
        super().run()
        root = Path(__file__).resolve().parent
        build = Path(self.build_lib).resolve().parent / "native"
        destination = Path(self.build_lib).resolve() / "torch_npu_memory_saver" / "lib"
        subprocess.run(
            [
                "cmake",
                "-S",
                str(root / "native"),
                "-B",
                str(build),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DSAVER_FAKE_ACL=OFF",
            ],
            check=True,
        )
        subprocess.run(["cmake", "--build", str(build), "--parallel"], check=True)
        destination.mkdir(parents=True, exist_ok=True)
        self.copy_file(str(build / "libascend_memory_saver.so"), str(destination))


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(cmdclass={"build_py": BuildPy}, distclass=BinaryDistribution)
