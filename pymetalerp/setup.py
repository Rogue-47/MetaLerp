import os
import subprocess
import numpy as np
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

"""
TODOS:
manylinux wheels
clang-linux compiler support
windows sdist and wheels support
"""
Version = "1.0"
Supported_Platforms = ["Linux x86_64"]

os.environ["CC"] = "gcc"
os.environ["CXX"] = "gcc"

CUDA_PATH = os.environ.get("CUDA_PATH", "/usr/local/cuda")
if not os.path.isdir(CUDA_PATH):
    print(f"CUDA_PATH {CUDA_PATH} not found. Please update CUDA_PATH and rerun.")
    exit(1)
if not os.path.isdir(os.path.join(CUDA_PATH, "include")):
    print("include directory not found in CUDA_PATH. Please update CUDA_PATH and try again.")
    exit(1)

resourceBase = os.path.join(os.path.dirname(__file__), "resources", "metalerp")

IncludeDirs = [
    np.get_include(),
    os.path.join(CUDA_PATH, "include"),
    resourceBase,
    ".",
]

LibDirs = [os.path.join(CUDA_PATH, "lib64"), os.path.dirname(__file__), resourceBase]

extraCompileArgs = [
    "-Ofast", "-ffast-math", "-fno-math-errno",
    "-funroll-loops", "-falign-functions=64",
    "-fprefetch-loop-arrays", "-msse4.2",
    "-fopenmp", "-fPIC",
    "-march=native", "-mtune=native", "-mavx", "-mavx2", "-mfma", "-flto",
]


Libs = ["m", "cudart", "cudadevrt", "gomp"]

extraLinkArgs = ["-fopenmp"]

longDescr = open(os.path.join(os.path.dirname(__file__), "resources", "README.md")).read()
longDescrType = "text/markdown"


def build_static_lib():
    srcDir = os.path.join(resourceBase, "core", "sources")
    cudaSrc = os.path.join(srcDir, "initializations.cu")
    cSrc = os.path.join(srcDir, "initializations.c")
    extInit = os.path.join(srcDir, "externals", "externals_init.c")

    outCuObj = os.path.join(os.path.dirname(__file__), "metalerp_cuda_runtime.o")
    outCObj = os.path.join(os.path.dirname(__file__), "metalerp_runtime.o")
    outExtObj = os.path.join(os.path.dirname(__file__), "ext_init.o")
    libOut = os.path.join(os.path.dirname(__file__), "libmetalerp.a")

    nvcc_bin = os.path.join(CUDA_PATH, "bin", "nvcc")
    if not (os.path.isfile(nvcc_bin) and os.access(nvcc_bin, os.X_OK)):
        print(f"nvcc not found or not executable at {nvcc_bin}. Please set CUDA_PATH correctly.")
        raise SystemExit(1)

    try:

        subprocess.check_call([
            nvcc_bin, "-DMETALERP_FAST", "-O3", "-use_fast_math",
            "-Xptxas=-O3,--disable-warnings",
            "--compiler-options", "-fPIC",
            "-Xcompiler", "-Ofast -ffast-math -fno-math-errno -mfma "
                          "-funroll-loops -falign-functions=64 -fprefetch-loop-arrays "
                          "-march=native -mtune=native -mavx -mavx2 -mf16c -msse4.2",
            "-gencode", "arch=compute_61,code=sm_61",
            "-gencode", "arch=compute_75,code=sm_75",
            "-gencode", "arch=compute_86,code=sm_86",
            "-gencode", "arch=compute_61,code=compute_61",
            "-c", "-o", outCuObj, cudaSrc
        ])


        subprocess.check_call([
            "gcc", "-fPIC", "-DMETALERP_INTERFACE_LIBMODE", "-DMETALERP_FAST",
            *extraCompileArgs, "-c", "-o", outCObj, cSrc
        ])
        subprocess.check_call([
            "gcc", "-fPIC", *extraCompileArgs, "-c", "-o", outExtObj, extInit
        ])


        subprocess.check_call([
            "ar", "rcs", libOut, outCObj, outCuObj, outExtObj
        ])


        for f in [outCuObj, outCObj, outExtObj]:
            try: os.remove(f)
            except FileNotFoundError: pass

    except subprocess.CalledProcessError as e:
        print("Static lib compilation failed:", e)
        raise SystemExit(1)


class FinalizeDependencies(build_ext):
    def run(self):
        build_static_lib()
        super().run()



extension = Extension(
    name="metalerp",
    sources=["metalerp.c"],
    include_dirs=IncludeDirs,
    library_dirs=LibDirs,
    libraries=Libs,  # cudart etc.
    extra_compile_args=extraCompileArgs,
    extra_link_args=extraLinkArgs + ["libmetalerp.a"],  
)


setup(
    name="metalerp",
    author="Omar M. Mahmoud",
    author_email="metalerplib@gmail.com",
    description="Fast transforms and approximations with CUDA-enabled processing routines.",
    long_description=longDescr,
    long_description_content_type=longDescrType,
    version=Version,
    platforms=Supported_Platforms,
    setup_requires=['numpy'],
    ext_modules=[extension],
    cmdclass={"build_ext": FinalizeDependencies},
    license="LGPL-v2.1",
    include_package_data=True,
    package_data={"": ["libmetalerp.a", "resources/metalerp/metalerp.h"]},
)
