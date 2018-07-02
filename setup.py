"""
sndtrck
"""

# from distutils.core import setup
from setuptools import setup

setup(
    name    = "sndtrck",
    version = "0.3.0",
    author  = "Eduardo Moguillansky",
    long_description = open("README.md").read(),
    description = "Sound manipulation and representation via partial-tracking",
    # installation
    packages = ["sndtrck"],
    package_data = {
        '': ["*.csd"],
        'sndtrck': ["*.csd"]
    }
)
