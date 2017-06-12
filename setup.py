"""
sndtrck


"""

# from distutils.core import setup
from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy as np

sndtrck_accel = Extension(
    'sndtrck.accel',
    sources = ['sndtrck/accel.pyx'],
    include_dirs = [np.get_include()]
)

setup(
    name     = "sndtrck",
    version  = "0.3.0",
    author   = "Eduardo Moguillansky",
    long_description = open("README.md").read(),
    description = "Sound manipulation and representation via partial-tracking",
    # installation
    packages = ["sndtrck"],
    cmdclass={'build_ext': build_ext},
    ext_modules=[sndtrck_accel]
)
