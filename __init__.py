#!/usr/bin/env python

# Author: Marek Rudnicki
# Time-stamp: <2010-02-25 15:52:03 marek>

# Description:

from __future__ import division
import numpy as np
from collections import namedtuple
import os

import neuron
from neuron import h

from gbc import GBC_Soma

lib_dir = os.path.dirname(__file__)
neuron.load_mechanisms(lib_dir)


EndbulbPars = namedtuple('EndbulbPars', ['e', 'tau', 'tau_fast',
                                         'tau_slow', 'U', 'k',
                                         'threshold', 'delay', 'weight'])




def main():
    import matplotlib.pyplot as plt
    import neuron
    import thorns.waves as wv

    h.celsius = 37
    tmax = 20

    gbc = GBC_Point()

    pars = EndbulbPars(e=0, tau=0.4, tau_fast=25, tau_slow=1000,
                       U=0.5, k=0.6, threshold=0, delay=0.5,
                       weight=0.016)
    train = np.array([8, 12])

    gbc.load_anf_train(train, pars)
    gbc.load_anf_train(train+1, pars)

    # clamp = h.SEClamp(gbc.soma(0.5))
    # clamp.amp1 = -65
    # clamp.dur1 = tmax
    # clamp.rs = 1

    # ci = h.Vector()
    # ci.record(clamp._ref_i)

    v = h.Vector()
    v.record(gbc.soma(0.5)._ref_v)

    print "Temperatur:", h.celsius
    neuron.init()
    gbc.init()
    neuron.run(tmax)

    plt.plot(wv.t(1/h.dt,v), v)
    plt.show()
    # plt.plot(wv.t(1/h.dt,ci), ci)
    # plt.show()


if __name__ == "__main__":
    main()
