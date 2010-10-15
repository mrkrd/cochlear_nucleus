from __future__ import division

__author__ = "Marek Rudnicki"


import numpy as np
from collections import namedtuple
import os

import neuron
from neuron import h

lib_dir = os.path.dirname(__file__)
neuron.load_mechanisms(lib_dir)

from gbc import GBC_Point

anf_type = [('typ', 'S3'),
            ('cf', float),
            ('spikes', np.ndarray)]


