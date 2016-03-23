from __future__ import division

__author__ = "Marek Rudnicki"

import os

import neuron
from neuron import h

lib_dir = os.path.dirname(__file__)
neuron.load_mechanisms(lib_dir)

from gbc import GBC_Point



def run(duration, objects=None):

    if objects is not None:
        for obj in objects:
            obj.pre_init()

    neuron.init()

    if objects is not None:
        for obj in objects:
            obj.post_init()

    neuron.run(duration*1000)



def set_celsius(celsius):
    h.celsius = celsius


def set_fs(fs):
    h.dt = 1000/fs
