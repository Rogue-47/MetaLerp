import os
import numpy
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


CUDA_PATH = os.environ.get("CUDA_PATH", "/usr/local/cuda")
if not os.path.isdir(CUDA_PATH):
    raise RuntimeError(f"CUDA_PATH {CUDA_PATH} not found.")

class BuildExtWithNumpy(build_ext):
    def finalize_options(self):
        
        super().finalize_options()
        self.include_dirs.append(numpy.get_include())
        self.include_dirs.append(os.path.join(CUDA_PATH, "include"))

ext = Extension(
    "metalerp",
    sources=["metalerp.c"],
    libraries=["metalerp", "cudart", "m"],
    library_dirs=[".", os.path.join(CUDA_PATH, "lib64")],
    define_macros=[("METALERP_FAST", None)],
    extra_compile_args=[
        "-Ofast", "-ffast-math", "-fno-math-errno",
        "-mfma", "-funroll-loops", "-falign-functions=64",
        "-fprefetch-loop-arrays", "-march=native", "-mtune=native",
        "-mavx", "-mavx2", "-mf16c", "-msse4.2",
        "-flto", "-fopenmp"
    ],
    extra_link_args=["-fopenmp"]
)

setup(
    name="metalerp",
    version="1.0",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtWithNumpy},
)
