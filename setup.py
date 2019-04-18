#!/usr/bin/env python

from setuptools import setup, find_packages

with open('README.rst') as file:
    long_description = file.read()

setup(
    name = "cochlear_nucleus",
    version = "2",
    author = "Marek Rudnicki",
    author_email = "marek.rudnicki@tum.de",

    description = "Models of cochlear nucleus neurons",
    license = "GPLv3+",
    url = "https://github.com/mrkrd/cochlear_nucleus",
    download_url = "https://github.com/mrkrd/cochlear_nucleus/tarball/master",

    packages = find_packages(),
    package_data = {
        "cochlear_nucleus.nrn": ["*.mod", ".csv"]
    },
    long_description = long_description,
    classifiers = [
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
    ],
)
