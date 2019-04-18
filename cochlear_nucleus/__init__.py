# -*- coding: utf-8 -*-

"""NEURON and Brian models of Globular Bushy Cells.

"""
from __future__ import division, print_function, absolute_import
from __future__ import unicode_literals

import sys

__version__ = "2"


_citation = """Rudnicki M. and Hemmert W. (2017).
High entrainment constrains synaptic depression levels of an in vivo
globular bushy cell model.
Frontiers in Computational Neuroscience, Frontiers Media SA, 2017, 11, pp. 1-11.
doi:10.3389/fncom.2017.00016
https://www.frontiersin.org/articles/10.3389/fncom.2017.00016/full"""

_citation_width = 80

print(
    " Please cite ".center(_citation_width, '='),
    file=sys.stderr
)
print(
    _citation,
    file=sys.stderr
)
print(
    " Thank you! ".center(_citation_width, '='),
    file=sys.stderr
)
