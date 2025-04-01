<h1 align="center">
<img src="https://github.com/sparrow-acoustics/sparrowpy/raw/main/docs/_static/logo.png" width="300">
</h1><br>


[![PyPI version](https://badge.fury.io/py/sparrowpy.svg)](https://badge.fury.io/py/sparrowpy)
[![Documentation Status](https://readthedocs.org/projects/sparrowpy/badge/?version=latest)](https://sparrowpy.readthedocs.io/en/latest/?badge=latest)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pyfar/gallery/main?labpath=docs/gallery/interactive/pyfar_introduction.ipynb)

Sound Propagation with Acoustic Radiosity for Realistic Outdoor Worlds

Getting Started
===============

The [pyfar workshop](https://mybinder.org/v2/gh/pyfar/gallery/main?labpath=docs/gallery/interactive/pyfar_introduction.ipynb)
gives an overview of the most important pyfar functionality and is a good
starting point. It is part of the [pyfar example gallery](https://pyfar-gallery.readthedocs.io/en/latest/examples_gallery.html)
that also contains more specific and in-depth
examples that can be executed interactively without a local installation by
clicking the mybinder.org button on the respective example. The
[pyfar documentation](https://pyfar.readthedocs.io) gives a detailed and complete overview of pyfar. All
these information are available from [pyfar.org](https://pyfar.org).

Installation
============

Use pip to install sparrowpy

    pip install sparrowpy

(Requires Python >= 3.9)

if numba is installed the code will be precompiled and will run faster. We strongly recommend to use numba to accelerate the simulations significantly

    pip install sparrowpy[fast]

or

    pip install sparrowpy
    pip install numba

you can also use blender for geometry import. Note that blender has strong requirements on the python version, see [pypi](https://pypi.org/project/bpy/). You can install it via pip

    pip install bpy

to show progress bars install tqdm

    pip install tqdm

by default these packages are not installed

Audio file reading/writing is supported through [SoundFile](https://python-soundfile.readthedocs.io), which is based on
[libsndfile](http://www.mega-nerd.com/libsndfile/). On Windows and OS X, it will be installed automatically.
On Linux, you need to install libsndfile using your distribution’s package manager, for example ``sudo apt-get install libsndfile1``.
If the installation fails, please check out the [help section](https://pyfar-gallery.readthedocs.io/en/latest/help).

Contributing
============

Check out the [contributing guidelines](https://pyfar.readthedocs.io/en/stable/contributing.html) if you want to become part of pyfar.
