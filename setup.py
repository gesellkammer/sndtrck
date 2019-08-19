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

with open('README.md') as f:
    long_description = f.read()


setup(
    name    = "sndtrck",
    version = get_version(),     # update sndtrck/version.py
    description = "Sound manipulation and representation via partial-tracking",
    author  = "Eduardo Moguillansky",
    long_description = long_description,
    long_description_content_type='text/markdown',
    # installation
    packages = ["sndtrck"],
    package_data = {
        '': ["*.csd"],
        'sndtrck': ["*.csd"]
    },
    install_requires = [
        "numpy",
        "bpf4>=0.7",
        "loristrck",
        "sounddevice",
        "pyqt5",
        "pyqtgraph",
        "notifydict",
        "appdirs",
        "pysdif3",
        "ctcsound",
        "pyliblo3",
        "emlib",
        "miniaudio"
    ]
)
