"""
sndtrck
"""

# from distutils.core import setup
from setuptools import setup

def get_version():
    d = {}
    with open("sndtrck/version.py") as f:
        code = f.read()
    exec(code, d)
    version = d.get('__version__', (0, 0, 0))
    return ("%d.%d.%d" % version).strip()

setup(
    name    = "sndtrck",
    version = get_version(),     # update sndtrck/version.py
    author  = "Eduardo Moguillansky",
    long_description = open("README.md").read(),
    description = "Sound manipulation and representation via partial-tracking",
    # installation
    packages = ["sndtrck"],
    package_data = {
        '': ["*.csd"],
        'sndtrck': ["*.csd"]
    },
    install_requires = [
        "numpy",
        "sndfileio>=0.6",
        "bpf4>=0.7",
        "loristrck",
        "sounddevice",
        "pyqt5",
        "pyqtgraph",
        "notifydict",
        "appdirs",
        "tinytag",
        "pysdif3",
        "ctcsound",
        "pyliblo3"
    ]
)
