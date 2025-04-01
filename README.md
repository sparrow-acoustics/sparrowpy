<h1 align="center">
<img src="https://github.com/sparrow-acoustics/sparrowpy/raw/main/docs/_static/logo.png" width="300">
</h1><br>

[![PyPI version](https://badge.fury.io/py/sparrowpy.svg)](https://badge.fury.io/py/sparrowpy)
[![Documentation Status](https://readthedocs.org/projects/sparrowpy/badge/?version=latest)](https://sparrowpy.readthedocs.io/en/latest/?badge=latest)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sparrow-acoustics/sparrowpy/main?urlpath=%2Fdoc%2Ftree%2Fexamples%2Ffast_radiosity.ipynb)
![pytest](https://github.com/sparrow-acoustics/sparrowpy/actions/workflows/tests.yml/badge.svg)
![ruff](https://github.com/sparrow-acoustics/sparrowpy/actions/workflows/ruff.yml/badge.svg)
![examples](https://github.com/sparrow-acoustics/sparrowpy/actions/workflows/test_examples.yml/badge.svg)

Sound Propagation with Acoustic Radiosity for Realistic Outdoor Worlds.

Getting Started
===============

Follow our [example notebooks](https://sparrowpy.readthedocs.io/en/latest/examples.html) to get a sense of the structure and functions of sparrowpy.

Installation
============

Use pip to install sparrowpy

    pip install sparrowpy

(Requires Python >= 3.9)

if numba is installed the code will be precompiled and will run faster. We strongly recommend to use numba to accelerate the simulations significantly

    pip install sparrowpy numba

Geometry import via blender or STL files will be supported in a future release, an install of the Blender API. Note that blender has strong requirements on the python version, see [pypi](https://pypi.org/project/bpy/). You can install it via pip

    pip install bpy

to show progress bars install tqdm

    pip install tqdm

by default these packages are not installed

Troubleshooting
===============

Audio file reading/writing in pyfar is supported through [SoundFile](https://python-soundfile.readthedocs.io), which is based on
[libsndfile](http://www.mega-nerd.com/libsndfile/). On Windows and OS X, it will be installed automatically.
On Linux, you need to install libsndfile using your distribution’s package manager, for example ``sudo apt-get install libsndfile1``.
If the installation fails, please check out the [help section](https://pyfar-gallery.readthedocs.io/en/latest/help).

Contributing
============

Check out the [contributing guidelines](https://sparrowpy.readthedocs.io/en/latest/contributing.html) if you want to become part of sparrowpy.
