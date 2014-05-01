#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name = "cochlear_nucleus",
    version = "0.1",
    packages = find_packages(),
    package_data = {
        "cochlear_nucleus.nrn": ["*.mod", ".csv"]
    },

    author = "Marek Rudnicki",
    author_email = "marek.rudnicki@tum.de",
    description = "Cochlear Nucleus neuron models in Python.",
    license = "GPL",
)
