import os
import numpy as np
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

Supported_Platforms = ["Linux x86_64"]

os.environ["CC"] = "gcc"
os.environ["CXX"] = "gcc"

if 'CUDA_PATH' in os.environ:
   CUDA_PATH = os.environ['CUDA_PATH']
else:
   print("Could not find CUDA_PATH in environment variables. Defaulting to /usr/local/cuda!")
   CUDA_PATH = "/usr/local/cuda"

if not os.path.isdir(CUDA_PATH):
   print(f"CUDA_PATH {CUDA_PATH} not found. Please update the CUDA_PATH variable and rerun")
   exit(0)

if not os.path.isdir(os.path.join(CUDA_PATH, "include")):
    print("include directory not found in CUDA_PATH. Please update CUDA_PATH and try again")
    exit(0)

IncludeDirs=[np.get_include(), os.path.join(CUDA_PATH, "include")]

LibDirs=[".", os.path.join(CUDA_PATH, "lib64")]


extraCompileArgs=[
        "-Ofast", "-ffast-math", "-fno-math-errno",
        "-funroll-loops", "-falign-functions=64",
        "-fprefetch-loop-arrays", "-msse4.2",
        "-fopenmp"
    ]

ext = Extension(
    "metalerp",
    sources=["metalerp.c"],
    libraries=["metalerp", "cudart", "m"],
    library_dirs=LibDirs,
    include_dirs=IncludeDirs,
    define_macros=[("METALERP_FAST", None)],
    extra_compile_args=extraCompileArgs,
    extra_link_args=["-fopenmp"]
)


class FinalizeDependencies(build_ext):
    def run(self):
        if os.environ.get("METALERP_NATIVE", "0") == "1":
            print("**********METALERP: Native build mode")
            for ext in self.extensions:
                ext.extra_compile_args.extend([
                    "-march=native", "-mtune=native", "-mavx", "-mavx2", "-flto", "-mfma"
                ])
        else:
            print("**********METALERP: Portable build mode")
        super().run()
    

setup(
    name="metalerp",
    author="Omar M. Mahmoud",
    author_email="metalerplib@gmail.com",
    download_url="https://github.com/Rogue-47/MetaLerp",
    description="Fast transforms and approximations provided by non-linear interpolation-based compute-friendly math kernels and dispatchers, with CUDA-enabled processing routines.",
    long_description="Check the README: https://github.com/Rogue-47/MetaLerp/blob/main/README.md",
    version="1.0",
    platforms=Supported_Platforms,
    setup_requires=['numpy'],
    ext_modules=[ext],
    cmdclass = {"build_ext" : FinalizeDependencies}
)
