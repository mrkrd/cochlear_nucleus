#!/usr/bin/env python

# Author: Marek Rudnicki
# Time-stamp: <2010-04-19 15:29:44 marek>

# Description:

from __future__ import division
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


