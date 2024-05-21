#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'numpy',
    'scipy',
    'matplotlib',
    'pyfar',
]

setup_requirements = [
    'pytest-runner',
]

test_requirements = [
    'pytest',
    'bump2version',
    'wheel',
    'watchdog',
    'ruff',
    'coverage',
    'Sphinx',
    'twine',
    'pydata-sphinx-theme',
]

setup(
    author="The pyfar developers",
    author_email='',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        
    ],
    description="Sound Propagation using Acoustic Radiosity in Python",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords='sparapy',
    name='sparapy',
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url="https://pyfar.org/",
    download_url="https://pypi.org/project/sparapy/",
    project_urls={
        "Bug Tracker": "https://github.com/ahms5/sparapy/issues",
        "Documentation": "https://sparapy.readthedocs.io/",
        "Source Code": "https://github.com/ahms5/sparapy",
    },
    version='0.1.0',
    zip_safe=False,
    python_requires='>=3.8',
)
