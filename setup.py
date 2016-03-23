#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name = "cochlear_nucleus",
    version = "1.1",
    author = "Marek Rudnicki",
    author_email = "marek.rudnicki@tum.de",

    description = "Models of cochlear nucleus neurons",
    license = "GPLv3+",

    packages = find_packages(),
    package_data = {
        "cochlear_nucleus.nrn": ["*.mod", ".csv"]
    },
)
