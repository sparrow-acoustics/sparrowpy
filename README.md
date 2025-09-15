<h1 align="center">
<p align="center">
    <img src="docs/_static/logo.svg" width="300"
             onerror="this.onerror=null;this.src='https://github.com/sparrow-acoustics/sparrowpy/raw/main/docs/_static/logo.svg';">
</p>
</h1><br>

[![PyPI version](https://badge.fury.io/py/sparrowpy.svg)](https://badge.fury.io/py/sparrowpy)
[![Documentation Status](https://readthedocs.org/projects/sparrowpy/badge/?version=latest)](https://sparrowpy.readthedocs.io/en/latest/?badge=latest)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sparrow-acoustics/sparrowpy/main?urlpath=%2Fdoc%2Ftree%2Fexamples%2Ffast_radiosity.ipynb)
![testing](https://github.com/sparrow-acoustics/sparrowpy/actions/workflows/pytest_pr.yml/badge.svg)

Sound Propagation with Acoustic Radiosity for Realistic Outdoor Worlds.

# Getting Started

Follow our [example notebooks](https://sparrowpy.readthedocs.io/en/stable/examples.html) to get a sense of the structure and functions of sparrowpy.

## Installation

Use pip to install sparrowpy

    pip install sparrowpy

(Requires Python >= 3.10)

if numba is installed the code will be precompiled and will run faster. We strongly recommend to use numba to accelerate the simulations significantly

    pip install sparrowpy numba

Geometry import via blender or STL files will be supported in a future release, an install of the Blender API. Note that blender has strong requirements on the python version, see [pypi](https://pypi.org/project/bpy/). You can install it via pip

    pip install bpy

by default these packages are not installed

## Contributing

Check out the [contributing guidelines](https://sparrowpy.readthedocs.io/en/stable/contributing.html) if you want to become part of sparrowpy.
