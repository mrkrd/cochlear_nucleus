from __future__ import division

__author__ = "Marek Rudnicki"

import os

import neuron
from neuron import h

lib_dir = os.path.dirname(__file__)
neuron.load_mechanisms(lib_dir)

from gbc import GBC_Point
